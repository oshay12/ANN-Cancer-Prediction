# isort was ran
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from keras import Input, layers
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, ReLU
from keras.models import Sequential
from matplotlib.ticker import MultipleLocator, PercentFormatter
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

# variables for CNN, I've found that these values work well with my model
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
OUT_CHANNELS = 64
BATCH_SIZE = 172
EPOCHS = 4

# loading data from .mat file, stores it in a dictionary
path = Path(__file__).resolve()
numRecogData = loadmat((path.parent / "NumberRecognitionBiggest.mat").resolve())


# helper function for preprocessing data for CNN
def numberRecogData(matFile):
    # putting data into numpy arrays so I can use numpy's features
    train = np.array(matFile["X_train"])
    labels = np.array(matFile["y_train"])
    test = np.array(matFile["X_test"])

    # normalizing training and testing RGB values so they function in the CNN,
    # and adding the channel information as the dataset does not provide it.
    # stolen from the keras_mnist file provided
    train = train.astype("float32") / 255
    test = test.astype("float32") / 255
    test = np.expand_dims(test, -1)

    # transposing and reshaping labels so it is of form (40000) rather than (1, 40000)
    labels = labels.transpose(1, 0).reshape(40000)

    # returns numbers and associated labels
    return train, labels, test


def question1(numberFile):
    # grabbing data from helper function
    train, labels = numberRecogData(numberFile)[0], numberRecogData(numberFile)[1]

    # initializing "blueprint" of CNN layers
    model = Sequential(
        [
            Conv2D(
                1,
                kernel_size=3,
                input_shape=(28, 28, 1),
                activation="linear",
                data_format="channels_last",
                padding="same",
                use_bias=True,
            ),
            ReLU(),
            BatchNormalization(),
            Flatten(),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    # compiles CNN from blueprint, specifies optimizer model, loss and metric functions for CNN
    model.compile(
        # had to use sparse categorical crossentropy as it is a multiclass classification task
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    # list of classifiers to loop through
    models = [
        model,
        KNeighborsClassifier(n_neighbors=1),
        KNeighborsClassifier(n_neighbors=5),
        KNeighborsClassifier(n_neighbors=10),
    ]

    # list of classifiers names
    classifiersLabeled = ["cnn", "knn1", "knn5", "knn10"]

    # initializing 2D lists of error rates, each interior list represents each model
    errorRates = [[], [], [], []]
    bestErrorRate = [0, 0, 0, 0]
    meanErrorRate = [0, 0, 0, 0]
    worstErrorRate = [0, 0, 0, 0]

    # initializing dataframe for kfold scores
    kfold_scores = pd.DataFrame(
        index=["cnn", "knn1", "knn5", "knn10"],
        columns=["fold1", "fold2", "fold3", "fold4", "fold5", "mean"],
        data=0.0,
    )

    kfold_scores["classifier"] = kfold_scores.index
    kfold_scores.set_index("classifier", inplace=True)
    kf_score = []

    # initializing dictionary to make graphing easier
    valuesDict = {
        "best": [0, 0, 0, 0],
        "mean": [0, 0, 0, 0],
        "worst": [0, 0, 0, 0],
    }

    # initializing K-Fold validation model, random state set so train-test
    # indicies are constant for each model
    StratKFold = StratifiedKFold(5, shuffle=True, random_state=40)

    # looping through each classifier, fitting them to training data,
    # cross-validating with the Stratified K-Fold model with 5 folds
    # and scoring with accuracy, then grabbing mean, best and
    # worst error rates
    for i in range(len(models)):
        # setting current model from classifiers list
        curClassifier = models[i]
        print("CURRENT CLASSIFIER: " + classifiersLabeled[i])

        # models require different shapes and properties of the input data, so I had to split the training and testing
        # of the two using if-else statements
        if classifiersLabeled[i] == "cnn":
            # model information
            curClassifier.summary()
            # cnn requires channel information, this adds the channel dimension
            cnnData = np.expand_dims(train, -1)

            # splitting data into 5 train-test splits using the stratified k-fold model for k-fold validation
            for j, (train_index, test_index) in enumerate(StratKFold.split(cnnData, labels)):
                print("FOLD #" + str(j + 1))
                # fitting model with train indicies of data for current fold
                curClassifier.fit(
                    cnnData[train_index],
                    labels[train_index],
                    batch_size=8,
                    epochs=1,
                    validation_split=0.1,
                )
                # predicting on test indicies of data for current fold
                score = curClassifier.evaluate(cnnData[test_index], labels[test_index], verbose=1)

                # grabbing error rate for current fold, adding to dataframe
                err = round(1 - score[1], 3)
                errorRates[i].append(err)
                kf_score = err
                kfold_scores.loc[classifiersLabeled[i], ("fold" + str(j + 1))] = kf_score

            # grabbing best, mean and worst error rate for model over the folds
            bestErrorRate[i] = min(errorRates[i])
            meanErrorRate[i] = np.mean(errorRates[i])
            worstErrorRate[i] = max(errorRates[i])
            valuesDict["best"][i] = bestErrorRate[i] * 100
            valuesDict["mean"][i] = meanErrorRate[i] * 100
            valuesDict["worst"][i] = worstErrorRate[i] * 100
            # adding mean error rate to dataframe
            kfold_scores.loc[classifiersLabeled[i], "mean"] = meanErrorRate[i]

        else:
            # knn requires flattened image data, so I'm reshaping it to fit the model
            knnData = train.reshape(40000, 784)

            # splitting data into 5 train-test splits using the stratified k-fold model for k-fold validation
            for j, (train_index, test_index) in enumerate(StratKFold.split(knnData, labels)):
                print("FOLD #" + str(j + 1))
                # fitting model with train indicies of data for current fold
                curClassifier.fit(knnData[train_index], labels[train_index])
                # predicting on test indicies of data for current fold
                preds = curClassifier.predict(knnData[test_index])

                # grabbing accuracy, calculating error and adding to dataframe
                score = accuracy_score(preds, labels[test_index])
                err = round(1 - score, 3)
                errorRates[i].append(err)
                kf_score = err
                kfold_scores.loc[classifiersLabeled[i], ("fold" + str(j + 1))] = kf_score

            # grabbing best, mean and worst error rate for model over the folds
            bestErrorRate[i] = min(errorRates[i])
            meanErrorRate[i] = np.mean(errorRates[i])
            worstErrorRate[i] = max(errorRates[i])
            valuesDict["best"][i] = bestErrorRate[i] * 100
            valuesDict["mean"][i] = meanErrorRate[i] * 100
            valuesDict["worst"][i] = worstErrorRate[i] * 100

            # adding mean error rate to dataframe
            kfold_scores.loc[classifiersLabeled[i], "mean"] = meanErrorRate[i]

    # saving kfold scores to json file
    save_kfold(kfold_scores, 1)

    sbn.set_style(style="darkgrid")

    # found code to make a faceted-esque bar plot for various models
    # using matplotlib. idea adapted from:
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    xAxis = np.arange(len(bestErrorRate))
    width = 0.25
    multiplier = 0

    fig, axis = plt.subplots(layout="constrained")

    # turning dict into list for easier access
    errors = list(valuesDict.items())
    for error, values in errors:
        offset = width * multiplier
        rects = axis.bar(xAxis + offset, values, width, label=error)
        axis.bar_label(rects, padding=3)
        multiplier += 1

    axis.set_ylabel("Error Rates")
    axis.set_title("Best, Mean and Worst Error Rates for Models (5 folds)")
    axis.set_xticks(xAxis + width, classifiersLabeled)
    axis.legend(loc="upper right", ncols=3)
    axis.set_ylim(0, 10)

    plt.gca().yaxis.set_major_formatter(PercentFormatter())  # set y axis to show percents
    plt.tight_layout()  # so all tick labels are shown properly
    plt.savefig("bonus1.png")
    # clear figure so no aesthetics get carried over to future plots
    plt.clf()


# loading csv into dataframe for q2 and q3 using pandas
breastCancerDF = pd.read_csv((path.parent / "breast-cancer.csv.xls").resolve())


# helper fuction for accessing the breast cancer dataframe's splits
def breastCancerData(df):
    # cleaning up data
    df = df.drop(["id"], axis=1)  # don't need id values

    # representing the labels numerically for the models
    df["diagnosis"] = df["diagnosis"].replace({"M": 1, "B": 0})
    # splitting df into just the features and just the labels
    testingData = df.drop(["diagnosis"], axis=1)
    labels = df["diagnosis"]

    # normalizing data in between 0 and 1
    # using sklearn's MinMaxScaler. adapted from:
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

    # looping through dataframe to normalize each column
    for i in testingData:
        testingData[i] = MinMaxScaler().fit_transform(testingData[i].values.reshape(-1, 1))

    return testingData, labels


# question 2 taken from assignment 2 as I am using the same dataset
def question2(bcFile):
    # grabbing data from helper function
    testingData, labels = breastCancerData(bcFile)

    # all feature names for breast cancer dataset
    FEAT_NAMES = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave_points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave_points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave_points_worst",
        "symmetry_worst",
        "fractal_dimension_worst",
    ]
    # column names in dataframe
    COLS = [
        "Feature",
        "AUC",
    ]
    # initialized dataframe for AUC values
    aucs = pd.DataFrame(
        columns=COLS,
        data=np.zeros([len(FEAT_NAMES), len(COLS)]),
    )

    for i, feat_name in enumerate(FEAT_NAMES):
        auc = roc_auc_score(y_true=labels, y_score=testingData.iloc[:, i])
        # fixing AUC values that are below 0.5
        if auc < 0.5:
            auc = 1 - auc
        aucs.iloc[i] = (feat_name, auc)

    # sorting by AUC values in descending order,
    # resetting the indicies to match the positions of the values
    aucs_sorted = aucs.sort_values(by="AUC", ascending=False).reset_index(drop=True)
    pd.DataFrame.to_json(aucs_sorted, (Path(__file__).resolve().parent / "aucs.json"))

    # grabbing top ten auc values from the dataframe
    topTenAUC = aucs_sorted.head(10)

    # setting size of figure
    plt.figure(figsize=(8, 6))
    plt.xlabel("AUC Scores")
    plt.tick_params(axis="y", which="major", labelsize=10)
    sbn.barplot(y=topTenAUC["Feature"], x=topTenAUC["AUC"], orient="horizontal").set(
        title="Ten Most Important Features shown through AUC values"
    )
    plt.xlim(0.75, 1.0)
    plt.xticks(np.arange(0.75, 1.0, 0.05))  # make AUC values tick up by 0.05
    plt.gca().set_axisbelow(True)  # make gridlines appear under bars
    plt.grid()
    plt.tight_layout()  # so the full feature names are shown
    plt.savefig("bonus2.png")
    # clear figure so no aesthetics get carried over to future plots
    plt.clf()


def question3(bcFile):
    # grabbing data from helper function
    data, labels = breastCancerData(bcFile)

    # initializing a few ANN's to test with different parameters
    ann1 = MLPClassifier(
        hidden_layer_sizes=(50),  # starting with 1 layer of 50 neurons
        solver="adam",
        activation="tanh",
        max_iter=1000,
        random_state=40,
        warm_start=True,  # reuses last call's solutions as initial input layer
        epsilon=1e-3,
    )

    ann2 = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # doubling hidden layer size, adding an extra layer of neurons
        solver="adam",
        activation="relu",  # switching to relu activation
        max_iter=1000,
        random_state=40,
        warm_start=True,
        alpha=0.00001,  # changing L2 regularization term's strength to 10x weaker than base
        epsilon=1e-12,
    )

    ann3 = MLPClassifier(
        hidden_layer_sizes=(200, 100, 50),  # doubling first layer again, adding 1 neuron layer
        solver="adam",
        activation="relu",
        max_iter=1000,
        random_state=40,
        warm_start=True,
        alpha=0.001,  # changing L2 regularization term's strength to 10x stronger than base
        epsilon=1e-2,
    )

    ann4 = MLPClassifier(
        hidden_layer_sizes=(400, 200, 100),  # doubled neurons for each layer compared to previous
        solver="adam",
        activation="identity",  # switching identity activation function
        max_iter=1000,
        random_state=40,
        warm_start=True,
        alpha=0.001,
        epsilon=1e-2,
    )

    # list of classifiers to loop through
    classifiers = [
        ann1,
        ann2,
        ann3,
        ann4,
        KNeighborsClassifier(n_neighbors=1),
        KNeighborsClassifier(n_neighbors=5),
        KNeighborsClassifier(n_neighbors=10),
    ]

    # names of classifiers
    classifiersLabeled = ["ann1", "ann2", "ann3", "ann4", "knn1", "knn5", "knn10"]

    # initializing 2D lists of error rates, each interior list represents each model
    errorRates = [[], [], [], [], [], [], []]
    bestErrorRate = [0, 0, 0, 0, 0, 0, 0]
    meanErrorRate = [0, 0, 0, 0, 0, 0, 0]
    worstErrorRate = [0, 0, 0, 0, 0, 0, 0]

    # initializing dataframe for kfold scores
    kfold_scores = pd.DataFrame(
        index=["ann1", "ann2", "ann3", "ann4", "knn1", "knn5", "knn10"],
        columns=["fold1", "fold2", "fold3", "fold4", "fold5", "mean"],
        data=0.0,
    )

    kfold_scores["classifier"] = kfold_scores.index
    kfold_scores.set_index("classifier", inplace=True)
    kf_score = []

    # initializing dictionary to make graphing easier
    valuesDict = {
        "best": [0, 0, 0, 0, 0, 0, 0],
        "mean": [0, 0, 0, 0, 0, 0, 0],
        "worst": [0, 0, 0, 0, 0, 0, 0],
    }

    # initializing dictionary for AUC values over folds
    aucDict = {}

    # initializing K-Fold validation model
    StratKFold = StratifiedKFold(5)

    # looping through each classifier, fitting them to training data,
    # cross-validating with the Stratified K-Fold model with 5 folds
    # and scoring with accuracy, then grabbing mean
    # and best score
    for i in range(len(classifiers)):
        # setting current classifier from list
        curClassifier = classifiers[i]
        print("CURRENT CLASSIFIER: " + classifiersLabeled[i])

        # initializing list to store auc values for each model
        aucs = []

        # splitting data into 5 train-test splits using the stratified k-fold model for k-fold validation
        for j, (train_index, test_index) in enumerate(StratKFold.split(data, labels)):
            print("FOLD #" + str(j + 1))
            # fitting models with data at the train indicies
            curClassifier.fit(data.iloc[train_index], labels.iloc[train_index])
            # evaulating models with data at the test indicies
            preds = curClassifier.predict(data.iloc[test_index])

            # grabbing accuracy, calculating error and adding to dataframe
            score = accuracy_score(preds, labels.iloc[test_index])
            auc = roc_auc_score(y_true=labels.iloc[test_index], y_score=preds)
            err = round(1 - score, 3)
            errorRates[i].append(err)
            aucs.append(auc)
            kf_score = err
            kfold_scores.loc[classifiersLabeled[i], ("fold" + str(j + 1))] = kf_score

        aucDict[classifiersLabeled[i]] = aucs
        # grabbing best, mean and worst error rate for model over the folds
        bestErrorRate[i] = min(errorRates[i])
        meanErrorRate[i] = np.mean(errorRates[i])
        worstErrorRate[i] = max(errorRates[i])
        valuesDict["best"][i] = bestErrorRate[i] * 100
        valuesDict["mean"][i] = meanErrorRate[i] * 100
        valuesDict["worst"][i] = worstErrorRate[i] * 100

        # adding mean error rate to dataframe
        kfold_scores.loc[classifiersLabeled[i], "mean"] = meanErrorRate[i]

    print(kfold_scores)
    save_kfold(kfold_scores, 3)

    # setting plots styles to seaborn's darkgrid
    sbn.set_style(style="darkgrid")

    # adapted code from question 1 to make facet-esque barplot of best,
    # mean and worst error rates for models
    xAxis = np.arange(len(bestErrorRate))
    width = 0.25
    multiplier = 0

    fig, axis = plt.subplots(layout="constrained")

    # turning dict into list for easier access
    errors = list(valuesDict.items())
    for error, values in errors:
        offset = width * multiplier
        rects = axis.bar(xAxis + offset, values, width, label=error)
        axis.bar_label(rects, padding=3)
        multiplier += 1

    axis.set_ylabel("Error Rates")
    axis.set_title("Best, Mean and Worst Error Rates for Models (5 folds)")
    axis.set_xticks(xAxis + width, classifiersLabeled)
    axis.legend(loc="upper left", ncols=3)
    axis.set_ylim(0, 10)

    plt.gca().yaxis.set_major_formatter(PercentFormatter())  # set y axis to show percents
    plt.tight_layout()  # so all tick labels are shown properly
    plt.savefig("bonus3.png")
    # clear figure so no aesthetics get carried over to future plots
    plt.clf()

    # setting plots styles to seaborn's darkgrid
    sbn.set_style(style="darkgrid")

    # setting figure size
    plt.figure(figsize=(10, 6))

    # looping through arrays inside the dictionary and plotting each models line
    for label, values in aucDict.items():
        plt.plot(range(1, len(values) + 1), values, marker="o", label=label)

    # setting tick locations and values using MultipleLocator
    x_locator = MultipleLocator(base=1.0)
    plt.gca().xaxis.set_major_locator(x_locator)
    y_locator = MultipleLocator(base=0.01)
    plt.gca().yaxis.set_major_locator(y_locator)

    plt.xlabel("Fold")
    plt.ylabel("AUC Score")
    plt.title("AUC Values Over Folds")
    plt.legend(loc="lower left", ncols=5)
    plt.savefig("bonus4.png")
    # clear figure so no aesthetics get carried over to future plots
    plt.clf()


# my attempt at creating the best CNN for the MNIST dataset
def cnnModel():
    model = Sequential(
        [  # specifying dimensions of input
            Input(shape=INPUT_SHAPE),
            # 2D convulution filter with 64 6x6 kernels and a filter shift, or stride, of 2x2 ran over images.
            # rectified linear unit activation function applied to solve the vanishing gradient
            # issue (makes the gradient that is passed to earlier functions in backpropagation non-vanishing)
            Conv2D(
                OUT_CHANNELS,
                kernel_size=(6, 6),
                strides=2,
                padding="same",
                data_format="channels_last",
                activation="relu",
            ),
            # adapted method from GoogLeNet's Inception V1 code, this reduces the
            # spatial dimensions of the data while retaining the important features using pooling,
            # very strong with image data. found at:
            # https://ai.plainenglish.io/googlenet-inceptionv1-with-tensorflow-9e7f3a161e87
            layers.MaxPooling2D(3, strides=2),
            # ensures that the activation functions are normalized for the layer
            BatchNormalization(),
            # process repeats for layer 2 of CNN
            Conv2D(
                OUT_CHANNELS,
                kernel_size=(6, 6),
                strides=2,
                padding="same",
                activation="relu",
            ),
            layers.MaxPooling2D(3, strides=2),
            BatchNormalization(),
            # data is flattened from a 3D vector to 1D, prepares it for dense layer
            Flatten(),
            # produces the final classification output, softmax activation is applied to
            # convert the models output to probabilities for each class
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    # summary of models layers, other model information
    model.summary()

    # compiling model
    model.compile(
        # had to use sparse categorical crossentropy as this is a multiclass classification problem
        loss="sparse_categorical_crossentropy",
        # after a lot of testing and research, adam seems to be the most reliable optimizer for this task
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


def question4(numberFile):
    # grabbing data from helper function
    train, labels, test = numberRecogData(numberFile)

    # initializing model
    model = cnnModel()

    kfold_scores = pd.DataFrame(
        index=["cnn"],
        columns=["fold1", "fold2", "fold3", "fold4", "fold5", "mean"],
        data=0.0,
    )

    kfold_scores["classifier"] = kfold_scores.index
    kfold_scores.set_index("classifier", inplace=True)
    kf_score = []
    scores = []

    # doing stratified k fold validaiton, fitting and evaulating model on training data
    StratKFold = StratifiedKFold(5)
    for j, (train_index, test_index) in enumerate(StratKFold.split(train, labels)):
        print("FOLD #" + str(j + 1))
        # training model for 4 epochs with batch size of 172
        model.fit(train[train_index], labels[train_index], batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)

        score = model.evaluate(train[test_index], labels[test_index])
        err = round(1 - score[1], 4)
        scores.append(err)
        kf_score = err
        kfold_scores.loc["cnn", ("fold" + str(j + 1))] = kf_score

    # getting mean value over folds
    mean = np.mean(round(err, 4))
    kfold_scores.loc["cnn", "mean"] = mean

    # scores for training data over folds
    print(kfold_scores)

    # predicting labels for test set
    pred = model.predict(test)
    # list of predictions for test set, argmax ran so regression values are changed to class labels
    y_pred = np.argmax(pred, axis=1)

    # saving predictions to predictions.npy file
    np.save(
        Path(__file__).resolve().parent / "predictions.npy",
        y_pred.astype(np.uint8),
        allow_pickle=False,
        fix_imports=False,
    )

    # saving kfold values to json file
    save_kfold(kfold_scores, 4)


def save_kfold(kfold_scores: pd.DataFrame, question: [1, 3, 4]) -> None:
    from pathlib import Path

    from pandas import DataFrame

    COLS = [*[f"fold{i}" for i in range(1, 6)], "mean"]
    INDEX = {
        1: ["cnn", "knn1", "knn5", "knn10"],
        3: ["knn1", "knn5", "knn10"],
        4: ["cnn"],
    }[question]
    outname = {
        1: "kfold_mnist.json",
        3: "kfold_data.json",
        4: "kfold_cnn.json",
    }[question]
    outfile = Path(__file__).resolve().parent / outname

    # name checks
    df = kfold_scores
    if not isinstance(df, DataFrame):
        raise ValueError("Argument `kfold_scores` to `save` must be a pandas DataFrame.")
    if kfold_scores.shape[1] != 6:
        raise ValueError("DataFrame must have 6 columns.")
    if df.columns.to_list() != COLS:
        raise ValueError(
            f"Columns are incorrectly named and/or incorrectly sorted. Got:\n{df.columns.to_list()}\n"
            f"but expected:\n{COLS}"
        )
    if df.index.name.lower() != "classifier":
        raise ValueError(
            "Index is incorrectly named. Create a proper index using `pd.Series` or "
            "`pd.Index` and use the `name='classifier'` argument."
        )
    idx_items = sorted(df.index.to_list())
    for idx in INDEX:
        if idx not in idx_items:
            raise ValueError(f"You are missing a row with index value {idx}")

    if question == 3:
        anns = df.filter(regex="ann", axis=0)
        if len(anns) < 2:
            raise ValueError(
                "You are supposed to experiment with different ANN configurations, "
                'but we found less than two rows with "ann" as the index name.'
            )

    if question == 3:
        df.to_json(outfile)
        print(f"K-Fold error rates for data successfully saved to {outfile}")
        return

    # value range checks
    if question == 1:
        if df.loc["cnn", "mean"] < 0.05:
            raise ValueError(
                "Your CNN error rate is too low. Make sure you implement the CNN as provided in "
                "the assignment or example code."
            )
        if df.loc[["knn1", "knn5"], "mean"].min() > 0.04:
            raise ValueError(
                "One of your KNN-1 or KNN-5 error rates is too high. There is likely an error in your code."
            )
        if df.loc["knn10", "mean"] > 0.047:
            raise ValueError("Your KNN-10 error rate is too high. There is likely an error in your code.")
        df.to_json(outfile)
        print(f"K-Fold error rates for MNIST data successfully saved to {outfile}")
        return

    # must be question 4
    df.to_json(outfile)
    print(f"K-Fold error rates for custom CNN on MNIST data successfully saved to {outfile}")


def main():
    question1(numRecogData)
    question2(breastCancerDF)
    question3(breastCancerDF)
    # question4(numRecogData)


if __name__ == "__main__":
    main()
