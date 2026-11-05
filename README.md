# Single-Layer Perceptron

A NumPy implementation of a single-layer perceptron with two inputs, two trainable weights, one bias, and one binary output. The notebook explores the OR, AND, and XOR truth tables and visualizes training boundaries, the final decision boundary, and sample-level errors.

## Architecture

For each input pair, the model computes:

```text
u = w[0] * x1 + w[1] * x2 + b
prediction = 0 if u < 0 else 1
```

There is no hidden layer. The step function is used during training. A sigmoid function is also defined and printed during interactive testing, but it is not used to train the model; its output should not be interpreted as a calibrated probability.

## Requirements

Use a Python environment with Jupyter and the following packages:

```bash
pip install numpy matplotlib seaborn pandas jupyter
```

Open the notebook in Jupyter Notebook, JupyterLab, or another compatible notebook environment.

## Training data

The four samples cover all possible binary input pairs:

```python
x1 = [0,0,1,1]
x2   = [0,1,0,1]
```

| x1 | x2 | OR | AND | XOR |
|---:|---:|---:|----:|----:|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 | 1 |
| 1 | 0 | 1 | 0 | 1 |
| 1 | 1 | 1 | 1 | 0 |

Choose **one** target list before building the DataFrame and training:

```python
Result = [0, 1, 1, 1]  # OR
# Result = [0, 0, 0, 1]  # AND
# Result = [0, 1, 1, 0]  # XOR
```

The notebook contains separate assignment cells for all three target lists. Running all of them in order leaves XOR as the active target because each assignment replaces the previous value. For OR or AND, execute only the desired target cell, then continue from the DataFrame cell.

OR and AND are linearly separable. A single-layer perceptron can learn them. XOR cannot be separated by one straight line, so this architecture cannot classify all four XOR samples correctly, regardless of how many training epochs are added.

## Code walkthrough

### 1. Import libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
```

NumPy handles numerical operations, Pandas creates the data table, and Seaborn and Matplotlib draw the plots.

### 2. Build the DataFrame and plot the data

```python
df = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'y': Result
})

sns.lmplot(
    data=df,
    x='x1',
    y='x2',
    hue='y',
    fit_reg=False,
    markers=['o', 's']
)

plt.show()
```

The horizontal axis is the first input, the vertical axis is the second input, and the color and marker distinguish target classes. `fit_reg=False` disables Seaborn's regression line. Use keyword arguments such as `data=`, `x=`, and `y=` for compatibility with the installed Seaborn interface.

Rebuild `df` whenever `Result` changes so that plotted labels match the selected task.

### 3. Define activation functions

```python
def step_function(x):
    if x<0:
        return 0
    else:
        return 1

def sigmoid_function(x):
  return 1 / (1 + np.exp(-x))
```

The step function returns 1 at zero. Only this function determines predictions in the training loop. The sigmoid is used later for comparison when testing a new input.

### 4. Initialize parameters

```python
#w = np.random.rand(2)
w = np.array([1,-1])
errors = []
eta = .90   # learning rate
epoch = 10
b = 0
```

The active initialization uses weights `[1, -1]`, bias `0`, learning rate `0.9`, and `10` epochs. The random initialization line is commented out. The training cell converts the weights to floating point so fractional updates are preserved.

### 5. Train and plot intermediate boundaries

```python
# وزن‌ها اعشاری باشند تا تغییرات اعشاری حفظ شوند
w = np.asarray(w, dtype=float)

for i in range(epoch):
    for x, y in zip(zip(x1, x2), Result):
        x = np.asarray(x, dtype=float)

        u = np.dot(x, w) + b
        error = y - step_function(u)
        errors.append(error)

        # به‌روزرسانی وزن‌ها
        for index, value in enumerate(x):
            w[index] += eta * error * value

        # یک بار برای هر نمونه
        b += eta * error

        plt.figure(1)

        if w[1] != 0:
            xs = np.array([min(x1), max(x1)])
            ys = -(w[0] * xs + b) / w[1]
            plt.plot(xs, ys, alpha=0.2)
        elif w[0] != 0:
            plt.axvline(-b / w[0], alpha=0.2)
```

Each epoch visits the four samples in a fixed order. For each sample:

1. Compute the weighted input plus bias.
2. Apply the step function and calculate `error = target - prediction`.
3. Append this signed error to `errors`.
4. Update each weight using its corresponding input.
5. Update the bias once for the sample, outside the weight-update loop.
6. Draw the updated decision boundary when one exists.

The learning rule is:

```text
w[j] = w[j] + eta * error * x[j]
b = b + eta * error
```

An error of zero leaves the parameters unchanged. The code always runs the specified number of epochs; it does not stop automatically after a mistake-free epoch.

The decision boundary satisfies `w[0] * x1 + w[1] * x2 + b = 0`. When the second weight is nonzero, it can be plotted as `x2 = -(w[0] * x1 + b) / w[1]`. When only the first weight is nonzero, the boundary is vertical. The branches avoid division by zero.

`plt.figure(1)` selects a figure by number. If that figure already contains another plot, the training lines will be added to it. Repeated execution can also accumulate lines. For an independent training plot, create a dedicated figure and axes before training and draw the intermediate boundaries on those axes.

### 6. Plot the final decision boundary

```python
# رسم نقاط داده
g = sns.lmplot(
    data=df,
    x='x1',
    y='x2',
    hue='y',
    fit_reg=False,
    markers=['o', 's']
)

ax = g.ax

# رسم مرز تصمیم: w[0]*x1 + w[1]*x2 + b = 0
if w[1] != 0:
    xs = np.array(ax.get_xlim())
    ys = -(w[0] * xs + b) / w[1]

    ax.plot(xs, ys, color='black', label='Decision boundary')

elif w[0] != 0:
    ax.axvline(
        x=-b / w[0],
        color='black',
        label='Decision boundary'
    )

else:
    print("هر دو وزن صفر هستند؛ مرز تصمیم قابل رسم نیست.")

ax.set_title('Final decision boundary')
ax.legend()
plt.show()
```

The final plot uses the last weights and bias. `g.ax` selects the axes belonging to the Seaborn plot so the line is drawn on top of the data. If both weights are zero, the prediction depends only on the bias and no separating line is drawn.

For XOR, “final” means the state after the last training step; it does not mean all samples have been classified correctly.

### 7. Plot training errors

```python
plt.figure(3)
plt.ylim([-1,1])
plt.plot(errors)
```

There are 40 recorded errors in a fresh run with four samples and ten epochs. The horizontal axis is the sample presentation index, not the epoch number. Each value is one of:

- `0`: the sample was classified correctly before that update.
- `1`: the target was 1 and the prediction was 0.
- `-1`: the target was 0 and the prediction was 1.

These are signed errors collected while the parameters are changing, not an epoch loss or a final accuracy score. Add `plt.show()` when an explicit display call is needed. A slightly wider vertical range, such as `plt.ylim(-1.2, 1.2)`, keeps the extreme values away from the plot edges.

### 8. Test an input interactively

```python
assign_input = int(input("enter Assignment number"))
exam_input = int(input("enter Exam number"))

input_numbers = (assign_input,exam_input)

print(input_numbers)

u1 = w[0]*assign_input + w[1]*exam_input + b
print(step_function(u1))
print(sigmoid_function(u1))
```

The prompt names refer to the earlier assignment/exam example. For the logic-gate experiment, enter 0 or 1 for each input. The first printed result is the input pair, followed by the binary step prediction and the sigmoid of the same raw score. No training occurs in this cell. Input validation is not implemented; non-integer text raises a conversion error.

## Running a fresh experiment

1. Execute the imports and input-data cells.
2. Select and execute one `Result` assignment.
3. Rebuild the DataFrame and plot the selected targets.
4. Define the activation functions.
5. Run parameter initialization to reset `w`, `b`, and `errors`.
6. Run training, then the final boundary and error plots.
7. Test binary input pairs using the interactive cell.

Before switching gates or comparing independent runs, reset the parameters and error list. Rerunning only the training cell continues from the current weights and bias and appends to the existing error history.

## Optional final evaluation

The following additional snippet evaluates all four samples using the final parameters without changing them:

```python
X = np.column_stack((x1, x2))
predictions = (X @ w + b >= 0).astype(int)
accuracy = np.mean(predictions == np.asarray(Result))

print("Targets:", Result)
print("Predictions:", predictions)
print("Training accuracy:", accuracy)
```

This measures accuracy on the training truth table. It is separate from the sample-level errors recorded during learning. For XOR, perfect accuracy is impossible with this two-input linear decision boundary.
