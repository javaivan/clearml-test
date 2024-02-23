import pandas as pd
from clearml import TaskTypes, Task
from clearml.automation.controller import PipelineDecorator


# Выгрузка данных
# Оборачиваем в декоратор
@PipelineDecorator.component(
    cache=False,  # Кешируем или нет
    return_values=['iris_data'],  # Название выходной переменной
    task_type=TaskTypes.custom,  # Тип таски
    name='load_data'  # название таски. По умолчанию - название фанкции
)
def load_iris():
    # Импортируем отдельно библиотеки, которые понадобятся в этом таске
    from sklearn import datasets
    import pandas as pd

    iris = datasets.load_iris()
    iris_data = pd.concat([pd.DataFrame(iris['data'], columns=iris['feature_names']), pd.DataFrame(iris['target'], columns=['target'])], axis=1)
    #     iris_data['target'] = iris_data['target'].apply(lambda x: 'setosa' if x==0 else ('versicolor' if x==1 else "virginica"))
    return iris_data


# Препроцессинг данных
# # Оборачиваем в декоратор
@PipelineDecorator.component(
    cache=False,  # Кешируем или нет
    return_values=["X_train_std", "y_train", "X_test_std", "y_test"],  # Название выходной переменной
    task_type=TaskTypes.data_processing,  # Тип таски
    name='preprocessing'  # название таски. По умолчанию - название фанкции
)
def preprocessing(
        data
):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = data.iloc[:, 2:4]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return (X_train_std, y_train, X_test_std, y_test)


# Обучение
# Оборачиваем в декоратор
@PipelineDecorator.component(
    cache=False,  # Кешируем или нет
    return_values=["model"],  # Название выходной переменной
    task_type=TaskTypes.training,  # Тип таски
    name='train_model'  # название таски. По умолчанию - название фанкции
)
def train_model(
        X_train,
        y_train,
):
    from sklearn.linear_model import Perceptron

    model = Perceptron(max_iter=40, eta0=0.1, random_state=1)
    model.fit(X_train, y_train)

    return model


# Постпроцессинг данных
# Оборачиваем в декоратор
@PipelineDecorator.component(
    cache=False,  # Кешируем или нет
    task_type=TaskTypes.custom,  # Тип таски
    return_values=["X_test", "acc"],  # Название выходной переменной
    name='postprocessing'  # название таски. По умолчанию - название фанкции
)
def postprocessing(
        X_train,
        X_test,
        y_train,
        y_test,
        model,
) -> pd.DataFrame:
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    import numpy as np
    from clearml import Task
    from sklearn.metrics import accuracy_score
    from datetime import datetime
    import pandas as pd

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0],
                        y=X[y == cl, 1],
                        alpha=0.8,
                        c=colors[idx],
                        marker=markers[idx],
                        label=cl,
                        edgecolor='black')

        if test_idx:
            X_test, y_test = X[test_idx, :], y[test_idx]

            plt.scatter(X_test[:, 0],
                        X_test[:, 1],
                        #                         c='',
                        edgecolor='black',
                        alpha=1.0,
                        linewidth=1,
                        marker='o',
                        s=100,
                        label='test set',
                        facecolor="none")

    X_combined_std = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    X_test = pd.DataFrame(X_test)

    X_test['predict'] = y_pred

    plot_decision_regions(X=X_combined_std, y=y_combined,
                          classifier=model, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.title(f'{datetime.now()}')
    plt.legend(loc='upper left')
    plt.show()

    return X_test, acc


# Главная функция работы пайплайна
@PipelineDecorator.pipeline(
    name='Iris_pipe',
    project='Test_pipe',
    version='0.2',
    return_value=["X_test_pred"]
)
def main(tags: list):
    from clearml import Task

    task = Task.current_task()
    #     task = Task.init(
    #         project_name="Test_pipe",
    #         task_name='Iris_pipe_2'
    #     )
    logger = task.get_logger()
    if tags is not None:
        task.add_tags(tags)
    data = load_iris()
    X_train, y_train, X_test, y_test = preprocessing(data=data)
    model = train_model(X_train=X_train, y_train=y_train)
    X_test_pred, accuracy = postprocessing(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, model=model)
    logger.report_single_value(name="accuracy", value=accuracy)


#     return X_test_pred

# if __name__ == "__main__":
# Запуск удаленно и очередью по дефолту
#     PipelineDecorator.set_default_execution_queue('default')
# # Запуск локально
PipelineDecorator.run_locally()
# Выполнение главной функции
## Эти агрументы идут в WEB ClearML
main(tags='Test_iris_tags_4')