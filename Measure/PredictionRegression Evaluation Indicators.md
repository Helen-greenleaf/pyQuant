

## Prediction/Regression Evaluation Indicators



Assume we have:

Prediction $$\hat{y} = {\hat{y}_1,\hat{y}_2, \dots \hat{y}_n }$$

Actual  $$y = {y_1,y_2, \dots y_n }$$



### Mean Square Error (MSE)



$$MSE = \frac{1}{n}\sum^n_{i=1}(\hat{y}_i - y_i)^2 $$

~~~Python
# import numpy as np
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred).dot(y_true - y_pred))

#from sklearn import metrics
#metrics.mean_squared_error(y_true, y_pred)
~~~



### Root Mean Square Error (RMSE)



$$RMSE = \sqrt{\frac{1}{n}\sum^n_{i=1}(\hat{y}_i - y_i)^2} $$

~~~Python
# import numpy as np
def mse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred).dot(y_true - y_pred)))

#from sklearn import metrics
#np.sqrt(metrics.mean_squared_error(y_true, y_pred))
~~~



### Mean Absolute Percentage Error (MAPE)


$$
MAPE=\frac{100\%}{n}\sum^n_{i=1}|\frac{\hat y_i-y_i}{y_i}|
$$

~~~Python
# import numpy as np
def mape(y_true, y_pred):
    return np.mean(np.sum(np.abs((y_pred - y_true) / y_true))) * 100
~~~



### Symmetric Mean Absolute Percentage Error (SMAPE)

$$ SMAPE = \frac{100\%}{n}\sum^n_{i=1} \frac{|\hat{y}_i- y_i|}{(|\hat{y}_i|+ |y_i|)^2} $$

~~~Python
# import numpy as np
def smape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)).dot(np.abs(y_pred) + np.abs(y_true))) * 100
~~~





### The sum of squares due to error (SSE)

$$
SSE=\sum^n_{i=1}(y_i-\hat y_i)^2
$$

```python
# import numpy as np
def SSE(y_true, y_pred):
    return (y_true-y_pred).dot(y_true-y_pred)
```





### $$R^2$$ Coefficient of determination

$$
R^2=1-\frac{SSE}{SST}=1-\frac{\sum^n_{i=1}{(y_i-\hat y_i)^2}}{\sum^n_{i=1}{(y_i-\bar y_i)^2 }}
$$

```python
# import numpy as np
def R_square(y_true, y_pred):
    return (1-((y_true - y_pred).dot(y_true - y_pred))/((y_true - np.mean(y_true)).dot(y_true - np.mean(y_true))))
```





### Degree-of-freedom adjusted $$R^2$$

The number of independent variables:   k
$$
\bar R^2=1-\frac{SSE/(n-k-1)}{SST/(n-1)}=1-(1-R^2)\frac{n-1}{n-k-1}
$$

```python
# import numpy as np
def Adjust_R_square(y_true, y_pred, k):
    R_square = (1-((y_true - y_pred).dot(y_true - y_pred))/((y_true - np.mean(y_true)).dot(y_true - np.mean(y_true))))
    return 1 - (1 - R_square)*(len(y_true) - 1)/(len(y_true) - k - 1)
```





###  Explained Variance Score


$$
explained\ variance\ score = 1-\frac{Var(y_i-\hat y_i)}{Var\ y_i}
$$

```python
# import numpy as np
def explained_variance_score(y_true, y_pred):
    return 1-np.var(y_true - y_pred)/np.var(y_true)
# from sklearn.metrics import explain_variance_score
# score = explained_variance_score(y_true, y_pred)
```



### Median absolute error


$$
MedianAE =\text{MedAE}(y, \hat{y}) = \text{median}(\mid y_1 – \hat{y}_1 \mid, \ldots, \mid y_n – \hat{y}_n \mid)
$$

```python
# import numpy as np
def medianAE(y_true, y_pred):
    return np.median(np.abs(y_true - y_pred))
# from sklearn.metrics import median_absolute_error
# median_absolute_error(y_true, y_pred)
```



### Mean squared logarithmic error


$$
MSLE = \frac1n\sum^n_{i=1}(\log(\hat y_i+1)-\log(y_i+1))^2
$$

```python
# import numpy as np
def msle(y_true, y_pred):
    return np.mean((np.log(y_pred + 1) - np.log(y_true + 1)).dot(np.log(y_pred + 1) - np.log(y_true + 1)))
```





### Confusion Matrix 

|            | y_pred = 0 | 1    |
| ---------- | ---------- | ---- |
| y_true = 0 | TN         | FP   |
| 1          | FN         | TP   |

```python
# import numpy as np
def confusion_matrix_two(y_true, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(0, len(y_true)):
        y = np.array([y_true[i], y_pred[i]])
        if y_true[i] == y_pred[i]:
            if np.any(y) == True:
                TP += 1
            else:
                TN += 1
        else:
            if y_pred[i] == 1:
                FP += 1
            else:
                FN += 1
    return np.array([TN, FP, FN, TP]).reshape(2,2)
#from sklearn.metrics import confusion_matrix
#confusion_matrix(y1_true,y1_pred)
```



- Accuracy
  $$
  accuracy = \frac{TP+TN}{TP+TN+FP+FN}
  $$
  

  ```python
  # import numpy as np
  def accuracy(y_true, y_pred):
    	cm = confusion_matrix_two(y_true, y_pred)
      return (cm[1,1] + cm[0,0])/np.sum(cm)
  ```

  

- Precision
  $$
  precision = \frac{TP}{TP+FP}
  $$

  ```python
  # import numpy as np
  def precision(y_true, y_pred):
    	cm = confusion_matrix_two(y_true, y_pred)
      return cm[1,1]/np.sum(cm[:,1])
  ```

  

  

- Recall

  $$
  recall=\frac{TP}{TP+FN}
  $$

  ```python
  # import numpy as np
  def recall(y_true, y_pred):
    	cm = confusion_matrix_two(y_true, y_pred)
      return cm[1,1]/np.sum(cm[1,:])
  ```

  

- F1 (H - mean)
  $$
  F1=\frac{2}{\frac{1}{precision}+\frac{1}{recall}}
  $$
  

  ```python
  # import numpy as np
  def F_1(y_true, y_pred):
      return 2/(1/precision(y_true, y_pred)+1/recall(y_true, y_pred))
  ```

  



###  Receiver Operating Characteristic Curve (ROC) 



### Area Under Curve (AUC)





