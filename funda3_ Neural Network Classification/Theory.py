"""
PyTorch Neural Network Classification
What is a classification problem?
    A classification problem involves predicting whether something is one thing or another.
    Classification = Predicting category / class label

    Examples:

        Email → Spam or Not Spam
        Tumor → Benign or Malignant
        Image → Cat / Dog / Horse
        Cricket shot → Pull / Cover Drive / Cut

        So output is discrete, not continuous.


Problem type	                   What is it?	                                                   Example
Binary classification	Target can be one of two options, e.g. yes or no	Predict whether or not someone has heart disease based on their health parameters.

Multi-class classification	Target can be one of more than two options	Decide whether a photo is of food, a person or a dog.

Multi-label classification	Target can be assigned more than one option	Predict what categories should be assigned to a Wikipedia article (e.g. mathematics, science & philosophy).
"""

"""
How Neural Network Does Classification

    A neural network: Input → Hidden Layers → Output Layer

    Mathematical Flow

    For one layer:

        z=Wx+b
        a=f(z)

    Where:

        W = weights
        b = bias
        f = activation function
        a = output

Output Layer Depends on Problem Type
    Binary Classification (2 classes)

        Input → Hidden Layers → 1 Output Neuron

        Example: Spam / Not Spam

        Output layer:  1 neuron

        Activation: Sigmoid

        Sigmoid converts output into probability:  σ(x)=1/1+e^−x​
        
        Output range:  0 to 1 → Probability

        If:
            0.5 → Class 1
            < 0.5 → Class 0

        Loss Function:

            Use:

                nn.BCEWithLogitsLoss()

         Best practice:

            Do NOT use sigmoid in forward
            Use BCEWithLogitsLoss (it applies sigmoid internally)

    Multi-class Classification (More than 2 classes)

        Input → Hidden → n Output Neurons

        Example: Cat / Dog / Horse

        Output layer: n neurons (equal to number of classes)

        Activation: Softmax

        Softmax(xi)= e^(xi)/∑e^(xj)

        If 3 classes → Output layer has 3 neurons.

            raw scores (logits)  ->  Output: [0.1, 0.7, 0.2]   -> Highest probability = predicted class
            
        Activation & Loss:

            Use: nn.CrossEntropyLoss()

            Important:

                DO NOT apply softmax manually.
                CrossEntropyLoss internally applies LogSoftmax.


    Multi-Label Classification
        Definition: Each sample can belong to multiple classes at the same time.

        Example:
            Wikipedia article:

                Mathematics ✅
                Science ✅
                Philosophy ❌

        So output might be: [1, 1, 0]

        Neural Network Structure: Input → Hidden → n Output Neurons

        But difference:

            Each output neuron is independent.
            We apply Sigmoid to each output neuron.

        Loss Function

            Use:
                nn.BCEWithLogitsLoss()
                Because each class is treated like separate binary classification.

Loss Function

For classification, we use:
    Binary → Binary Cross Entropy (BCE)
        Loss=−[ylog(p)+(1−y)log(1−p)]

    Multi-class → Cross Entropy Loss
        nn.CrossEntropyLoss()
        Do NOT apply softmax manually beacuse CrossEntropyLoss already applies LogSoftmax internally


| Feature    | Regression       | Classification    |
| ---------- | ---------------- | ----------------- |
| Output     | Continuous value | Category          |
| Activation | None / Linear    | Sigmoid / Softmax |
| Loss       | MSE, L1          | CrossEntropy      |
| Example    | House price      | Spam detection    |



"""
"""
Hyperparameter                               | Binary Classification                                           | Multiclass Classification
---------------------------------------------------------------------------------------------------------------
Input layer shape (in_features)       | Same as number of features                            | Same as binary classification
                                                            | (e.g. 5 for age, sex, height, etc.)                       |

Hidden layer(s)                                  | Problem specific (min = 1, unlimited possible) | Same as binary classification

Neurons per hidden layer                 | Problem specific (generally 10–512)                 | Same as binary classification

Output layer shape (out_features)    | 1 (single output neuron)                                     | 1 per class 
                                                              |                                                                              | (e.g. 3 for food, person, dog)

Hidden layer activation                     | Usually ReLU                                                       | Same as binary classification

Output activation                               | Sigmoid                                                                 |Softmax

Loss function                                     | Binary Cross Entropy                                          | Cross Entropy
                                                           | (BCELoss / BCEWithLogitsLoss preferred)        | (CrossEntropyLoss)

Optimizer                                           | SGD, Adam, etc.                                                  | Same as binary classification

"""

