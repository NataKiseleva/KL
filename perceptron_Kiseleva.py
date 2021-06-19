
from typing import Union, List
from math import sqrt
class Scalar:
  pass
class Vector:
  pass

class Scalar:
  def __init__(self: Scalar, val: float):
    self.val = float(val)

  def __mul__(self: Scalar, other: Union[Scalar, Vector]) -> Union[Scalar, Vector]:
    if isinstance(other, Vector):
      res = []
      for i in range(len(other.entries)):
        res.append(self.val * other.entries[i])
      return Vector(*res)
    elif isinstance(other, Scalar):
        return Scalar(self.val * other.val)
    else:
      print("No")

  def __add__(self: Scalar, other: Scalar) -> Scalar:
    return Scalar(self.val + other.val)

  def __sub__(self: Scalar, other: Scalar) -> Scalar:
    return Scalar(self.val - other.val)

  def __truediv__(self: Scalar, other: Scalar) -> Scalar:
    return Scalar(self.val / other.val)

  def __rtruediv__(self: Scalar, other: Vector) -> Vector:
    res = []
    for i in range(len(other)):
      x = other.entries[i]/self.val
      res.append(x)
    return Vector(*res)

  def __repr__(self: Scalar) -> str:
    return "Scalar(%r)" % self.val

  def sign(self: Scalar) -> int:
    if self.val == 0:
      return 0
    elif self.val < 0:
      return -1
    else:
      return 1

  def __float__(self: Scalar) -> float:
    return self.val

class Vector:
  def __init__(self: Vector, *entries: List[float]):
    self.entries = entries

  def zero(size: int) -> Vector:
    return Vector(*[0 for i in range(size)])

  def __add__(self: Vector, other: Vector) -> Vector:
    if len(self.entries) == len(other):
      res = []
      for i in range(len(self.entries)):
        res.append(self.entries[i] + other.entries[i])
    return Vector(*res)

  def __sub__(self: Vector, other: Vector) -> Vector:
    if len(self.entries) == len(other):
      res = []
      for i in range(len(self.entries)):
        res.append(self.entries[i] - other.entries[i])
    return Vector(*res)

  def __mul__(self: Vector, other: Vector) -> Scalar:
    res_v = 0
    if len(self.entries) == len(other.entries):
      for i in range(len(self.entries)):
        res_v += self.entries[i] * other.entries[i]
    return Scalar(res_v)

  def magnitude(self: Vector) -> Scalar:
    res = 0
    for i in range(len(self.entries)):
      res += self.entries[i]**2
    return Scalar(sqrt(res))

  def unit(self: Vector) -> Vector:
    return self / self.magnitude()

  def __len__(self: Vector) -> int:
    return len(self.entries)

  def __repr__(self: Vector) -> str:
    return "Vector%s" % repr(self.entries)
    
  def __iter__(self: Vector):
    return iter(self.entries)

def PerceptronTrain(D, maxiter = 100):
  w = Vector.zero(len(D[0][0]))
  b = Scalar(0)

  for i in range(maxiter):
    for x, y in D:
      a = x*w + b
      if (y*a).sign() <= 0: 
        w += y*x
        b += y
  return w, b

def PerceptronTest(w, b, D):
  res = []
  for x,y in D:
    a = x*w + b
    res.append(a.sign())
  return res

from random import randint

v = Vector(randint(-100, 100), randint(-100, 100))
xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys = [v * x * Scalar(randint(-1, 9)) for x in xs]

def train_test(data1, data2, split):

  b = int(len(data1)*split)
  data = []

  for i in range(len(data1)):
    data.append((data1[i], data2[i]))

  train = data[:b]
  test= data[b:]

  return train, test

train, test = train_test(xs, ys, 0.9)

w1, b1 = PerceptronTrain(train)
y_pred = PerceptronTest(w1, b1, test)

def score(y_pred, y_true):
  all = len(y_true)
  correct = 0
  for i in range(all):
    if y_pred[i] == y_true[i][1].sign():
      correct += 1
  return correct/all*100

score(y_pred, test)

xs_xor = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys_xor = [Scalar(1) if x.entries[0]*x.entries[1] < 0 else Scalar(-1) for x in xs_xor]

train_xor, test_xor = train_test(xs_xor, ys_xor, 0.9)

w2, b2 = PerceptronTrain(train_xor)
y_pred_xor = PerceptronTest(w2, b2, test_xor)

score(y_pred_xor, test_xor)

train_for_exp = train

train_sorted = sorted(train, key= lambda x: x[1].val)

performance = []

for i in range(1, 11):
  w, b = PerceptronTrain(train_sorted, maxiter=i)
  y_pr = PerceptronTest(w, b, test)
  performance.append(score(y_pr, test))

performance_1 = []

for i in range(1, 11):
  shuffle(train_for_exp)
  w, b = PerceptronTrain(train_for_exp, maxiter=i)
  y_pr = PerceptronTest(w, b, test)
  performance_1.append(score(y_pr, test))

def PerceptronTrainPermute(D, maxiter = 100):
  w = Vector.zero(len(D[0][0]))
  b = Scalar(0)

  for i in range(maxiter):
    shuffle(D)
    for x, y in D:
      a = x*w + b
      if (y*a).sign() <= 0: 
        w += y*x
        b += y
  return w, b

performance_2 = []

for i in range(1, 11):
  w, b = PerceptronTrainPermute(train_for_exp, maxiter=i)
  y_pr = PerceptronTest(w, b, test)
  performance_2.append(score(y_pr, test))

import matplotlib.pyplot as plt

_ = plt.plot(list(range(1, 11)), performance, marker='d', linestyle='dashed', color = 'gray', label = 'no permutation')
_ = plt.plot(list(range(1, 11)), performance_1, marker='d', linestyle='dashed', color = 'blue', label = 'random permutation at the beginning')
_ = plt.plot(list(range(1, 11)), performance_2, marker='d', linestyle='dashed', color = 'green', label = 'random permutation at each epoch')
_ = plt.xlabel('Epochs', fontsize=12)
_ = plt.ylabel('Score', fontsize=12)
_ = plt.legend()
_ = plt.title('different strategies', fontsize=14)
plt.show()

def AveragedPerceptronTrain(D, maxiter = 100):
  w = Vector.zero(len(D[0][0]))
  b = Scalar(0)
  u = Vector.zero(len(D[0][0]))
  beta = Scalar(0)
  c = Scalar(1)
  for i in range(maxiter):
    shuffle(D)
    for x, y in D:
      a = x*w + b
      if (y*a).sign() <= 0: 
        w += y*x
        b += y
        u += y*c*x
        beta += y*c
      c += Scalar(1)
  return w-(Scalar(1)/c)*u, b-beta*(Scalar(1)/c)

performance_3 = []

for i in range(1, 11):
  w, b = AveragedPerceptronTrain(train, maxiter=i)
  y_pr = PerceptronTest(w, b, test)
  performance_3.append(score(y_pr, test))

_ = plt.plot(list(range(1, 11)), performance_2, marker='d', linestyle='dashed', color = 'green', label = 'PerceptronTrain')
_ = plt.plot(list(range(1, 11)), performance_3, marker='d', linestyle='dashed', color = 'gray', label = 'AveragedPerceptron')
_ = plt.xlabel('Epochs', fontsize=12)
_ = plt.ylabel('Score', fontsize=12)
_ = plt.legend(fontsize=11)
_ = plt.title('AveragedPerceptron versus PerceptronTrain', fontsize=14)
plt.show()
