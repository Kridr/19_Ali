from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
import time
rs = 1
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rs)

scaler = StandardScaler()
transformer = PowerTransformer()

clf = GradientBoostingClassifier(n_estimators=300, random_state=rs)

start = time.perf_counter()
clf.fit(X_train, y_train)
print('Fit time no preprocessing:', time.perf_counter() - start)
print(clf.score(X_test, y_test))

X_train = scaler.fit_transform(X_train)
start = time.perf_counter()
clf.fit(X_train, y_train)
print('Fit time standardization:', time.perf_counter() - start)
print(clf.score(X_test, y_test))

X_train = transformer.fit_transform(X_train)
start = time.perf_counter()
clf.fit(X_train, y_train)
print('Fit time normalization:', time.perf_counter() - start)
print(clf.score(X_test, y_test))

