from gmab import Gmab, tester, GmabSearchCV
import random

def test_function(number: list) -> float:
    return sum([i ** 2 for i in number])

def rosenbrock_function(number: list):
    return sum([100 * (number[i + 1] - number[i] ** 2) ** 2 + (1 - number[i]) ** 2 for i in range(len(number) - 1)])

def random_noise_rosenbrock_function(number: list):
    return rosenbrock_function(number) + 0.1 * (2 * random.random() - 1)


if __name__ == '__main__':
    bounds = [(-5, 10), (-5, 10)]
    gmab = Gmab(random_noise_rosenbrock_function, bounds)
    evaluation_budget = 10000
    result = gmab.optimize(evaluation_budget)
    print(result)

    tester()

    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import uniform
    iris = load_iris()
    logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200,
                                  random_state=0)
    distributions = dict(C=uniform(loc=0, scale=4))
    clf = GmabSearchCV(logistic, distributions)
    search = clf.fit(iris.data, iris.target)
    print(search.best_params_)
