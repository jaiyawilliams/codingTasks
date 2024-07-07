# codingTasks

# Titanic (3).ipynb
In this task, a Jupyter notebook named titanic.ipynb was created to perform an in-depth Exploratory Data Analysis (EDA) on the Titanic dataset. The analysis began with loading the dataset and necessary libraries, followed by an initial data overview to understand its structure and contents. Missing values were handled by filling or dropping as appropriate, and data cleaning steps were performed to ensure consistency and accuracy. New features were engineered to provide additional insights. The EDA addressed specific guiding questions, such as identifying key factors influencing survival, analyzing the preference given to upper-class passengers and women and children, and uncovering any additional noteworthy observations. Various visualizations, such as bar plots and pair plots, were used to illustrate findings, and detailed explanations accompanied each step to enhance understanding. The task concluded with summarizing key insights derived from the analysis, ensuring a comprehensive understanding of the dataset's implications.

![image](https://github.com/jaiyawilliams/codingTasks/assets/166830191/4486f8c3-fa59-4ae9-af92-135d81307956)

![image](https://github.com/jaiyawilliams/codingTasks/assets/166830191/a8fde6e8-39d0-4713-990f-6645c2c27460)

![image](https://github.com/jaiyawilliams/codingTasks/assets/166830191/7447de84-b2f2-43fe-928f-37a5f7feef9c)

![image](https://github.com/jaiyawilliams/codingTasks/assets/166830191/f1f93920-98e0-49e5-a2c4-d98882c0459b)

![image](https://github.com/jaiyawilliams/codingTasks/assets/166830191/eed15086-727e-4cfd-9b29-ee7583fd40e0)





# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Mnist_task.ipynb
In this task, the Jupyter notebook file MNIST.ipynb was duplicated and renamed to mnist_task.ipynb. The objective was to perform a comprehensive analysis of the MNIST dataset. The dataset was loaded using the load_digits function from the sklearn.datasets module. The data was then split into training and testing sets, with an explanation provided on the importance of these sets in evaluating model performance. A Random Forest Classifier from scikit-learn was employed to create a classification model. One parameter of the model was chosen for tuning, with a rationale given for the selection and its specific value during testing. The confusion matrix for the model's performance on the test set was printed, highlighting which classes the model struggled with the most. Additionally, the task involved reporting the model's accuracy, precision, recall, and F1-score, using the macro average method in scikit-learn's metrics functions. This analysis aimed to provide a detailed understanding of the classifier's performance and areas for improvement.

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns



# Neural_networks_task.ipynb
In this task, a copy of the neural_network.ipynb file was created and renamed to neural_network_task.ipynb. The notebook guided the user through a series of exercises to model basic logic gates using neural networks. Specifically, it involved determining the values needed to model an AND gate, a NOR gate, and a NAND gate. Then, the task required combining these gates to create an XOR gate, which more closely resembles a neural network structure. This involved calculating the appropriate shapes for the weights and biases. The practical task aimed to provide hands-on experience with neural network concepts by applying them to fundamental logical operations.

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

![image](https://github.com/jaiyawilliams/codingTasks/assets/166830191/7faae013-9818-46fc-ac31-e6575c28d356)

