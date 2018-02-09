# Linear Modelling - Least Squares

_Linear modelling_ is the process of learning a linear relationship between attributes and responces. Consider the following data set where $x\_n$ denotes the nth olympic year and $t\_n$ denotes the nth time of the mens 100m.

```python
{"cmd":"echo \"$CODE\" >> res/mens100m.py"}
x = [1896, 1900, 1904, 1906, 1908, 1912, 1920, 1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 
     1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008]
t = [12.00, 11.00, 11.00, 11.20, 10.80, 10.80, 10.80, 10.60, 10.80, 10.30, 10.30, 10.30, 10.40, 10.50, 
     10.20, 10.00, 9.95, 10.14, 10.06, 10.25, 9.99, 9.92, 9.96, 9.84, 9.87, 9.85, 9.69]
```

```python
{"cmd":"echo \"$CODE\" | python"}
from res import mens100m
import matplotlib.pyplot as plt

plt.plot(mens100m.x, mens100m.t, 'ro')
plt.savefig("res/graph1.svg")
```

![](res/graph1.svg)

## Defining the model

We can clearly see a statistical dependence between year and time (note this is not clausal).

We will define are model as a function that maps the input attributes (olympic year) to the target values (winning time). Clearly this function will involves terms other than $x$ called _parameters_, these are the values that we learn from the data set. A function $f$ that maps input attributes $x\_0, ..., x\_n$ with parameters $w\_0, ..., w\_k$ is denoted as $f(x\_0, ..., x\_n; w_0, ... w_k)$.

To pick which model we are going to use, we make the assumtion that the data can be modelled linearly, thus the function will be of the form.

$$
t = f(x; w_0, w_1) = w_0 + w_1 x
$$

## Loss function

To find the _best_ model we must define what the _best_ means. In our case we a looking for the line which is as close to all data points as possible. A common way to measure this is the squared diffrence known as the squared loss function:

$$
\mathcal{L}_n(t_n, f(x_n; w_0, w_1)) = (t_n - f(x_n; w_0, w_1))^2
$$

The smaller the loss for year $n$, the closer the model at $x_n$ is to $t_n$. We need a low loss for all years, thus we find the average loss:

$$
\mathcal{L} = \frac{1}{N} \sum^{N}_{n = 1} L_n(t_n, f(x_n; w_0, w_1))
$$

Thus we will tune $w_0$ and $w_1$ such that the average loss is minimilized, expressed mathmatically:

$$
\operatorname*{argmin}_{w_0, w_1} = \frac{1}{N} \sum^N\_{n=1} L_n(t_n, f(x_n; w_0, w_1)) 
$$

We have choosen this loss function because we can find values for $w_0$ and $w_1$ analytically however other loss functions could be used such as the absolute loss:

$$
\mathcal{L}_n = \left| t_n - f(x_n; w_0, w_1) \right|
$$

## Finding a solution

To find the $\operatorname*{argmin}_{w_0, w_1} = \frac{1}{N} \sum^N\_{n=1} L_n(t_n, f(x_n; w_0, w_1))$ we can differentiate to find the point in which the gradient is zero (the minima of the function).

First we substiture the linear model into the expression for average loss:
$$
\begin{align}
\mathcal{L} &= \frac{1}{N} \sum^{N}_{n = 1} L\_n(t\_n, f(x\_n; w\_0, w\_1)) \\\\
&= \frac{1}{N} \sum^{N}\_{n = 1} (t\_n - f(x\_n; w\_0, w\_1))^2 \\\\
&= \frac{1}{N} \sum^{N}\_{n = 1} (t\_n - (w\_0 x\_n + w\_1))^2 \\\\
&= \frac{1}{N} \sum^{N}\_{n = 1} (w\_1^2 x\_n^2 + 2 w\_1 x\_n w\_0 - 2 w\_1 x\_n t\_n + w\_0^2 - 2 w\_0 t\_n + t\_n^2) \\\\
&= \frac{1}{N} \sum^{N}\_{n = 1} (w\_1^2 x\_n^2 + 2 w\_1 x\_n (w\_0 - t\_n) + w\_0^2 - 2 w\_0 t\_n + t\_n^2) \\\\
\end{align}
$$

Differentiate with respect to $w\_0$

$$
\begin{align}
\frac{1}{N} \sum^{N}\_{n = 1} (w\_0^2 + 2 w\_1 x\_n w\_0 - 2 w\_0 t\_n) && \text{Remove terms not including $w\_0$} \\\\
w\_0^2 + 2 w\_1 w\_0 \frac{1}{N} \left( \sum^{N}\_{n = 1} x\_n \right) - 2 w\_0 \frac{1}{N} \left( \sum^{N}\_{n = 1} t\_n \right) && \text{Rearrange} \\\\
\frac{\partial{\mathcal{L}}}{\partial{w\_0}} = 2 w\_0 + 2 w\_1 \frac{1}{N} \left( \sum^{N}\_{n = 1} x\_n \right) - \frac{2}{N} \left( \sum^{N}\_{n = 1} t\_n \right) && \text{Differentiate}
\end{align}
$$

Differentiate with respect to $w\_1$

$$
\begin{align}
\frac{1}{N} \sum^{N}\_{n = 1} (w\_1^2 x\_n^2 + 2 w\_1 x\_n w\_0 - 2 w\_1 x\_n t\_n)) && \text{Remove terms not including $w\_1$} \\\\
w\_1^2\frac{1}{N} \left( \sum^{N}\_{n = 1} x\_n^2 \right) + 2 w\_1 w\_0 \frac{1}{N} \left( \sum^{N}\_{n = 1} x\_n \right) - 2 w\_1 \frac{1}{N} \left( \sum^{N}\_{n = 1} x\_n t\_n \right) && \text{Rearrange} \\\\
\frac{\partial{\mathcal{L}}}{\partial{w\_1}} = 2 w\_1 \frac{1}{N} \left( \sum^{N}\_{n = 1} x\_n^2 \right) + 2 w\_0 \frac{1}{N} \left( \sum^{N}\_{n = 1} x\_n \right) - \frac{2}{N} \left( \sum^{N}\_{n = 1} x\_n t\_n \right) && \text{Differentiate}
\end{align}
$$

Differentiate again, to check if the turning point is a minima
$$
\begin{align}
\frac{\partial^2{\mathcal{L}}}{\partial{w\_0}} &= 2 \\\\
\frac{\partial^2{\mathcal{L}}}{\partial{w\_1}} &= \frac{2}{N} \sum^N\_{n=1}{x^2\_n} \\\\
\end{align}
$$


Set $\frac{\partial{\mathcal{L}}}{\partial{w\_0}}$ to $0$

$$
\begin{align}
2 w\_0 + 2 w\_1 \frac{1}{N} \left( \sum^{N}\_{n = 1} x\_n \right) - \frac{2}{N} \left( \sum^{N}\_{n = 1} t\_n \right) &= 0 \\\\
2 w\_0 &= \frac{2}{N} \left( \sum^{N}\_{n = 1} t\_n \right) - 2 w\_1 \frac{1}{N} \left( \sum^{N}\_{n = 1} x\_n \right) \\\\
w\_0 &= \frac{1}{N} \left( \sum^{N}\_{n = 1} t\_n \right) - w\_1 \frac{1}{N} \left( \sum^{N}\_{n = 1} x\_n \right) \\\\
\widehat{w\_0} &= \bar{t} - w\_1 \bar{x} \\\\
\end{align}
$$

Subing in the new expression for $w\_0$ into $\frac{\partial{\mathcal{L}}}{\partial{w\_1}}$ we get:

$$
\begin{align}
\frac{\partial{\mathcal{L}}}{\partial{w\_1}} &= 2 w\_1 \frac{1}{N} \left( \sum^{N}\_{n = 1} x\_n^2 \right) + 2 \hat{w\_0} \frac{1}{N} \left( \sum^{N}\_{n = 1} x\_n \right) - \frac{2}{N} \left( \sum^{N}\_{n = 1} x\_n t\_n \right) \\\\
&= 2 w\_1 \frac{1}{N} \left( \sum^{N}\_{n = 1} x\_n^2 \right) + 2 (\bar{t} - w\_1 \bar{x}) \frac{1}{N} \left( \sum^{N}\_{n = 1} x\_n \right) - \frac{2}{N} \left( \sum^{N}\_{n = 1} x\_n t\_n \right) \\\\
&= w\_1 \frac{2}{N} \left( \sum^{N}\_{n = 1} x\_n^2 \right) + \bar{t} \frac{2}{N} \left( \sum^{N}\_{n = 1} x\_n \right) - w\_1 \bar{x} \frac{2}{N} \left( \sum^{N}\_{n = 1} x\_n \right) - \frac{2}{N} \left( \sum^{N}\_{n = 1} x\_n t\_n \right)
\end{align}
$$

Simplify using $\bar{x} = \frac{1}{N} \sum^{N}\_{n = 1} x\_n$

$$
\frac{\partial{\mathcal{L}}}{\partial{w\_1}} = w\_1 \frac{2}{N} \left( \sum^{N}\_{n = 1} x\_n^2 \right) + 2 \bar{t} \bar{x} - 2 w\_1 (\bar{x})^2 - \frac{2}{N} \left( \sum^{N}\_{n = 1} x\_n t\_n \right)
$$

Set $\frac{\partial{\mathcal{L}}}{\partial{w\_1}}$ to $0$

$$
\begin{align}
w\_1 \frac{2}{N} \left( \sum^{N}\_{n = 1} x\_n^2 \right) + 2 \bar{t} \bar{x} - 2 w\_1 (\bar{x})^2 - \frac{2}{N} \left( \sum^{N}\_{n = 1} x\_n t\_n \right) &= 0 \\\\
w\_1 \frac{2}{N} \left( \sum^{N}\_{n = 1} x\_n^2 \right) - 2 w\_1 (\bar{x})^2 &= \frac{2}{N} \left( \sum^{N}\_{n = 1} x\_n t\_n \right) - 2 \bar{t} \bar{x} \\\\
w\_1 \left[\frac{2}{N} \left( \sum^{N}\_{n = 1} x\_n^2 \right) - 2 (\bar{x})^2 \right] &= \frac{2}{N} \left( \sum^{N}\_{n = 1} x\_n t\_n \right) - 2 \bar{t} \bar{x} \\\\
w\_1 &= \frac{\frac{1}{N} \left( \sum^{N}\_{n = 1} x\_n t\_n \right) - \bar{t} \bar{x}}{\frac{1}{N} \left( \sum^{N}\_{n = 1} x\_n^2 \right) - (\bar{x})^2} \\\\
\end{align}
$$

To simplify we will define a few more average quantitys, $\overline{x^2} = \frac{1}{N} \sum^{N}\_{n = 1} x\_n^2$ and $\overline{xt} = \frac{1}{N} \sum^{N}\_{n = 1} x\_n t\_n$, thus we arrive at:

$$
\widehat{w\_1} = \frac{\overline{xt} - \bar{x} \bar{t}}{\overline{x^2} - (\bar{x})^2}
$$

Thus all that we need to compute the best parameter values are:

$$
\widehat{w\_0} = \bar{t} - w\_1 \bar{x} \quad\quad\quad
\widehat{w\_1} = \frac{\overline{xt} - \bar{x} \bar{t}}{\overline{x^2} - (\bar{x})^2}
$$

## Least Square Fit

```python
{"cmd":"echo \"$CODE\" | python"}

import numpy as np
import matplotlib.pyplot as plt
from res import mens100m

x, t = mens100m.x, mens100m.t
t_bar = np.average(t)
x_bar = np.average(x)
xt_bar = np.average([x[i] * t[i] for i in range(len(x))])
x2_bar = np.average(list(map(lambda xi: xi**2, x)))

w1 = (xt_bar - x_bar * t_bar) / (x2_bar - x_bar ** 2)
w0 = t_bar - w1 * x_bar

print("w0 = {}, w1 = {}".format(w0, w1))

model_x = np.linspace(min(x), max(x), 100)
model_t = list(map(lambda xi: w0 + w1 * xi, model_x))

plt.plot(x, t, 'ro')
plt.plot(model_x, model_t)
plt.savefig("res/graph2.svg")
```

![](res/graph2.svg)

## Extending to vectors

Clearly their are lots of attributes which affect mens 100m times. We can extend are model by using vectors, consider the following:

$$
\mathbf{X} = \begin{bmatrix}1 & x_1 \\\\ 1 & x_2 \\\\ \vdots & \vdots \\\\ 1 & x_n\end{bmatrix}
\quad
\mathbf{t} = \begin{bmatrix}t_1 \\\\ t_2 \\\\ \vdots \\\\ t_n\end{bmatrix}
\quad
\mathbf{w} = \begin{bmatrix}w_0 \\\\ w_1\end{bmatrix}
$$

The loss can be written as:

$$
\mathcal{L} = \frac{1}{N}(\mathbf{t}-\mathbf{Xw})^T (\mathbf{t}-\mathbf{Xw})
$$

Which we can see by expanding out:

$$
\begin{align}
\mathcal{L} &= \frac{1}{N} \times \left( \begin{bmatrix}t_1 - w_0 - w_1 x_1 \\\\ t_2 - w_0 - w_1 x_2 \\\\  \vdots  \\\\  t_n - w_0 - w_1 x_n \end{bmatrix} \right)^T \times \left( \begin{bmatrix}t_1 - w_0 - w_1 x_1 \\\\ t_2 - w_0 - w_1 x_2 \\\\  \vdots  \\\\  t_n - w_0 - w_1 x_n \end{bmatrix} \right) \\\\ 
&= \frac{1}{N} \left[ (t_1 - w_0 + w_1 x_1)^2 + ... + (t_n - w_0 + w_1 x_n)^2 \right]  \\\\ 
&= \frac{1}{N} \sum^{N}_{n = 1} L_n(t_n, f(x_n; w_0, w_1))
\end{align}
$$

As before we differentiate with respect to $\mathbf{w}$ (the parameters in vector form), to do this we can use the following rules:

| $f(\mathbf{w})$ | $\frac{\partial f}{\partial \mathbf{w}}$ |
| :-------------: |:-------------:|
| $\mathbf{w}^T \mathbf{x}$ | $\mathbf{x}$ |
| $\mathbf{x}^T \mathbf{w}$ | $\mathbf{x}$ |
| $\mathbf{w}^T \mathbf{w}$ | $2 \mathbf{w}$ |
| $\mathbf{w}^T \mathbf{Cw}$ | $2 \mathbf{Cw}$ |

Thus we get the following:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{2}{N} \mathbf{X}^T \mathbf{Xw} - \frac{2}{N} \mathbf{X}^T \mathbf{t} 
$$

To find $\widehat{\mathbf{w}}$ (the optimum parameters) we set $\frac{\partial \mathcal{L}}{\partial \mathbf{w}}$ and use inverses since division isnt defined for matricies.

$$
\begin{align}
\frac{2}{N} \mathbf{X}^T \mathbf{Xw} - \frac{2}{N} \mathbf{X}^T \mathbf{t} &= 0 \\\\
\frac{2}{N} \mathbf{X}^T \mathbf{Xw} &= \frac{2}{N} \mathbf{X}^T \mathbf{t} \\\\
\mathbf{X}^T \mathbf{Xw} &= \mathbf{X}^T \mathbf{t} \\\\
\mathbf{I} \mathbf{w} &= (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{t} \\\\
\widehat{\mathbf{w}} &= (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{t} \\\\
\end{align}
$$

Now we can define linear models with any number of parameters.

## Non-linear responces

So far the model takes the form $f(x; \mathbf{w}) = w_0 + w_1 x$, however by extending matrix $\mathbf{X}$ we get a polynomial function of any order:

$$
\mathbf{X} = \begin{bmatrix}
x_1^0 & x_1^1 & \cdots & x_1^K \\\\
x_2^0 & x_2^1 & \cdots & x_2^K \\\\
\vdots & \vdots & \ddots & \vdots \\\\
x_N^0 & x_N^1 & \cdots & x_N^K \\\\
\end{bmatrix}
$$

Once more we can extend this past polynomial function with $K$ functions of $x$, $h_k(X)$:

$$
\mathbf{X} = \begin{bmatrix}
h_1(x_1) & h_2(x_1) & \cdots & h_K(x_1) \\\\
h_1(x_2) & h_2(x_2) & \cdots & h_K(x_2) \\\\
\vdots & \vdots & \ddots & \vdots \\\\
h_1(x_N) & h_2(x_N) & \cdots & h_K(x_N) \\\\
\end{bmatrix}
$$

```python
{"cmd":"echo \"$CODE\" >> res/exampleModels.py"}

import numpy as np

# Original linear model
def f1(x, w0, w1):
    return w0 + np.multiply(w1, x)

# Second order polinomial
def f2(x, w0, w1, w2):
    return w0 + np.multiply(w1, x) + np.multiply(w2, np.power(x, 2))

# 5-th order polinomial
def f3(x, w0, w1, w2, w3, w4, w5):
    return w0 + np.multiply(w1, x) + np.multiply(w2, np.power(x, 2)) + np.multiply(w3, np.power(x, 3)) \
         + np.multiply(w4, np.power(x, 4)) + np.multiply(w5, np.power(x, 5))

# With sin
def f4(x, w0, w1, w2, a, b):
    return w0 + np.multiply(w1, x) + np.multiply(w2, np.power(x, 2)) + np.multiply(w1, np.sin((x - a)/b))

functions = [
    (f1, 'r', "First order polynomial (linear)"), 
    (f2, 'g', "Second order polynomial"), 
    (f3, 'y', "5th order polynomial"), 
    (f4, 'b', "Function with sin term")
]
```

```python
{"cmd":"echo \"$CODE\" | python"}

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from res import mens100m, exampleModels

x, t = mens100m.x, mens100m.t
model_x = np.linspace(min(x), max(x), 100)

i = 0
for (f, color, title) in exampleModels.functions:
    plt.figure()
    popt, pcov = curve_fit(f, x, t)
    plt.plot(x, t, 'ro')
    plt.plot(model_x, f(model_x, *popt), color)
    plt.title(title)
    plt.savefig("res/graph3-{}.svg".format(i))
    i += 1
```

![](res/graph3-0.svg)
![](res/graph3-1.svg)
![](res/graph3-2.svg)
![](res/graph3-3.svg)

## Generalization and overfitting

The question becomes, what is the best model. We want a model that _generalize_ beyond the training data to make future predictions. Clearly the 5th order polynomial has a lower loss that lower order polynomials, however it doesnt make good future predictions since it pays to much attention to the training data, and is _overfitted_.

```python
{"cmd":"echo \"$CODE\" | python"}

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from res import mens100m, exampleModels

x, t = mens100m.x, mens100m.t
model_x = np.linspace(min(x), max(x), 100)

plt.plot(x, t, 'ro')

# training range
popt, pcov = curve_fit(exampleModels.f3, x, t)
plt.plot(model_x, exampleModels.f3(model_x, *popt), 'y')

# prediction range
pred_x = np.linspace(max(x), max(x) + 40, 40)
popt, pcov = curve_fit(exampleModels.f3, x, t)
plt.plot(pred_x, exampleModels.f3(pred_x, *popt), 'y--')

plt.title("5th order polynomial")
plt.savefig("res/graph4.svg")
```

![](res/graph4.svg)

Determining the optimum model complexity such that it generalizes well without overfitting is refered to as the bias-variance tradeoff and is very challenging.

## Validation data

A common way to overcome this problem is splitting the dataset into training data and validation data. Several models are trained with the training data, then the loss is computed for each against the validation data. Models that generalize well will have a lower loss against the validation data, where as models that are overfitted will have higher losses.

```python
{"cmd":"echo \"$CODE\" | python"}

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from res import mens100m, exampleModels

x, t = mens100m.x, mens100m.t
model_x = np.linspace(min(x), max(x), 100)

training_x = x[:-10]
training_t = t[:-10]

validation_x = x[-10:]
validation_t = t[-10:]

i=0
for (f, color, title) in exampleModels.functions:
    plt.figure()
    popt, pcov = curve_fit(f, training_x, training_t)
    plt.plot(training_x, training_t, 'ro')
    plt.plot(validation_x, validation_t, 'rs')
    plt.plot(model_x, f(model_x, *popt), color)
    plt.title(title)
    plt.savefig("res/graph5-{}.svg".format(i))
    i += 1
```

![](res/graph5-0.svg)
![](res/graph5-1.svg)
![](res/graph5-2.svg)
![](res/graph5-3.svg)

## K-Fold cross validation

The loss for the validation data is sensitive to how we split the dataset, cross validation is a tecnique that allows us to make more efficent use of the data.

K-Fold cross validation splits the dataset into $K$ blocks, each block in turn is used as validation data and the other $K -1$ blocks as training data. The final loss value is the average of all $K$ loss values.

The extream case where $K = N$ is known as leave one out cross validation (LOOCV), which can be expressed as the following:
$$
\mathcal{L}^{CV} = \frac{1}{N} \sum\_{n=1}^N (t\_n - \widehat{\mathbf{w}}\_{-n}^T x\_n)^2
$$
Where $\widehat{\mathbf{w}}\_{-n}^T$ is the estimate of the parameters without the $n$th datapoint.

## Regularized least squares

Both types of validation prevents models from becoming too complex, another solution is regularization. In general the higher the values of the parameters, the more complex the model becomes, thus instead of average squared loss, we can consider the regularized loss:

$$
\mathcal{L}' = \mathcal{L} + \lambda \mathbf{w}^T \mathbf{w}
$$

The term $\lambda \mathbf{w}^T \mathbf{w}$ penalises complexity with parameter $\lambda$ controlling the trade off between not fitting the data well and penalising complexity. To determine a value for $\lambda$ that gives the best predictive performance we can use cross validation.