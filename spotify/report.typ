#import "template.typ": *
#import "@preview/cetz:0.1.2": canvas, plot, palette, draw

#let legend-item(point, name, style) = {
  draw.content(
    (point.at(0)+2, point.at(1)), (point.at(0)+2, point.at(1)), 
    frame: "rect",
    padding: .3em,
    fill: style.fill,
    stroke: none,
    [],
  )
  draw.content(
    (point.at(0)+2.4, point.at(1)+0.1), (point.at(0)+5.5, point.at(1)),
    [
      #v(.2em)
      #text(name, size: .7em, weight: "bold")
    ]
  )
}

#show: project.with(
  title: "Machine Learning with Spotify dataset",
  authors: (
    (name: "Marcus Lee", email: "marcustutorial@hotmail.com"),
  ),
  date: "November 22, 2023",
  font: "CMU Serif",
  monofont: "CMU Typewriter Text",
)

= Individual classifiers vs Ensemble (vote) of classifiers 

== Evaluation measure(s)

I chose *F1-Measure* because it is a combination of both precision and recall, hence it reflects both the true positive rate and false positive rate of the classifiers' performance which is why this measure depicts a more complete picture of each classifier's strengths and weaknesses compared to other measures such as accuracy. Hence, it is easier for us to compare and determine the trade-offs between classifiers. \ \

To clarify, Weka labels it as `F-Measure` but it is the same as F1-Measure 
because it uses harmonic mean ($beta = 1$) in the calculation of:

$ F = frac((1 + beta^2) dot.c text("Precision") dot.c text("Recall"), beta^2 dot.c text("Precision") + text("Recall")) $

Note that in Weka, the weighted average of F-Measure was calculated by:

$ F_("weighted avg") = frac(sum_(k=1)^n F_k dot.c "count(k)", sum_(k=1)^n "count(k)") $

where $n$ is the number of classes, $F_k$ is the F-Measure of class $k$ and $"count"(k)$ is the number of instances of class $k$.

== Evaluation of the three classifiers

The weka classifiers used for the evaluation are:

#list(
  indent: 1em,
  [*Decision Tree* $arrow.r$ `weka.classifiers.trees.J48`],
  [*Neural Network* $arrow.r$ `weka.classifiers.functions.MultiLayerPerceptron`],
  [*k-NN* $arrow.r$ `weka.classifiers.lazy.IBk`]
)

#align(center)[
  #figure(
    table(
      columns: (auto, auto, auto, auto),
      inset: 6pt,
      align: horizon,
      fill: (col, row) => if row == 0 or row == 6 or col == 0 { silver } else { white },
      [], [*Decision Tree*], [*Neural Network*], [*k-NN (k=1)*],
      [*edm*], [0.593], [0.608], [0.542],
      [*latin*], [0.408], [0.375], [0.411],
      [*pop*], [0.343], [0.360], [0.331],
      [*rap*], [0.608], [0.629], [0.549],
      [*rock*], [0.653], [0.679], [0.608],
      [*Weighted Average*], [*0.522*], [*0.532*], [*0.488*],
    ),
    caption: [F1-Measure of the three classifiers on the dataset]
  )
]

The table above shows the results of the classifiers, they were ran with default classifier settings and test options of 10-fold cross validation. In this case, k-fold cross validation was preferred over train test split so that the classifiers are trained on as much data as possible and because the dataset is small, the incurred computational overhead is acceptable. 

As can be seen from the results, the _neural network_ classifier performed the best with an average F1-Measure of *0.532*. The _decision tree_ classifier performed the second best with an average F1-Measure of *0.522*. The _k-NN_ classifier performed the worst with an average F1-Measure of *0.488*. One of the possible cause that neural network performed the best because it is capable of learning complex non-linear relationship across the features set. 

== Evaluation of the ensemble of the three classifiers

The combination rules used for the ensemble are:

#list(
  indent: 1em,
  [*Average of Probabilities*],
  [*Majority Voting*],
  [*Minimum Probability*]
)

=== Results

To ensure the fairness of the test, the ensemble was also ran with 10-fold cross validation. The results for different combination rules are as follows:

#align(center)[
  #figure(
    table(
      columns: (auto, auto, auto, auto),
      inset: 6pt,
      align: horizon,
      fill: (col, row) => if row == 0 or row == 6 or col == 0 { silver } else { white },
      [], [*Average of Possibilities*], [*Majority Voting*], [*Minimum Probability*],
      [*edm*], [0.614], [0.621], [0.540],
      [*latin*], [0.435], [0.432], [0.402],
      [*pop*], [0.366], [0.378], [0.345],
      [*rap*], [0.644], [0.650], [0.551],
      [*rock*], [0.696], [0.703], [0.618],
      [*Weighted \ Average*], [*0.552*], [*0.557*], [*0.491*],
    ),
    caption: [F1-Measure of the ensemble of the three classifiers]
  )
]

Looking at the table above, we can see that _Majority Voting_ produces the best result with an average F1-Measure of *0.557*. The _Average of Probabilities_ produces the second best result with an average F1-Measure of *0.552*. The _Minimum Probability_ produces the worst result with an average F1-Measure of *0.491*.

=== Results justification 

_Majority Voting_ performed the best because it utilises a simple voting scheme and since the dataset has discrete class labels such as `edm`, `latin`, `pop`, `rap` and `rock`, it is robust to noisy prediction wherein even a few classifiers produced incorrect predictions the influence on the final decision is minor as long as majority of the classifiers produced correct predictions. 

As for _Minimum Probability_, it performed the worst due to its conservativeness in which it only focuses on the minimum probability. This is especially not ideal for this dataset because the dataset is small hence a lot of information was discarded when solely looking at the minimum probability. 

The justification for _Average of Possibilities_ is that although it does not perform as well as _Majority Voting_, it is a safe go-to approach since it factors in both the strengths and weaknesses of all classifiers by averaging their output.

In conclusion, comparing the overall results of the ensemble with the individual classifiers, we can see that the ensemble performed better than the individual classifiers. This is because the ensemble is able to combine the strengths of the individual classifiers and mitigate their weaknesses.

= Ensemble with bagging

== Results with increasing bag size

The configuration for the results are as follows:
#list(
  indent: 1em,
  [*Decision Tree* $arrow.r$ `weka.classifiers.trees.J48 (10-fold cross validation`)],
  [*Neural Network* $arrow.r$ `weka.classifiers.functions.MultiLayerPerceptron (80/20 train test split`)],
  [*k-NN* $arrow.r$ `weka.classifiers.lazy.IBk (10-fold cross validation)`]
)

The results below were ran using `weka.classifiers.meta.Bagging` with the above configuration and increasing `numIterations` (equivalent to bag size) from $2$ to $20$. Note that, neural network does not use 10-fold cross validation because it takes too long to run. For the sake of simplicity, only the results for _Weighted Average_ is included.

#let bg_dtree = (
  (2, 0.495),
  (4, 0.548),
  (6, 0.563),
  (8, 0.572),
  (10, 0.579),
  (12, 0.583),
  (14, 0.584),
  (16, 0.587),
  (18, 0.588),
  (20, 0.592)
)
#let bg_nn = (
  (2, 0.559),
  (4, 0.560),
  (6, 0.565),
  (8, 0.565),
  (10, 0.566),
  (12, 0.566),
  (14, 0.567),
  (16, 0.566),
  (18, 0.567),
  (20, 0.566)
)
#let bg_knn = (
  (2, 0.467), 
  (4, 0.484), 
  (6, 0.489), 
  (8, 0.492), 
  (10, 0.492), 
  (12, 0.494), 
  (14, 0.494), 
  (16, 0.494), 
  (18, 0.495), 
  (20, 0.495)
)

#align(center)[
  #figure(
    table(
      columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
      inset: 6pt,
      align: horizon,
      fill: (col, row) => if row == 0 or col == 0 { silver } else { white },
      [], [*2*], [*4*], [*6*], [*8*], [*10*], [*12*], [*14*], [*16*], [*18*], [*20*],
      [*Decision Tree (J48)*], ..bg_dtree.map(it => [#it.at(1)]),
      [*Neural \ Network*], ..bg_nn.map(it => [#it.at(1)]),
      [*k-NN*], ..bg_knn.map(it => [#it.at(1)])
    ),
    caption: [F1-Measure of the ensemble (bagging) with increasing bag size]
  )
]

#align(center)[
  #figure(
    canvas(length: 1cm, {
      plot.plot(
        size: (8, 6),
        x-tick-step: 2,
        x-label: "Bag Size",
        y-min: 0.4,
        y-max: 0.6,
        y-tick-step: 0.1,
        y-label: "Weighted F1-Measure",
        {
          plot.add(
            mark: "o",
            mark-size: .1,
            style: (mark: (stroke: blue)),
            bg_dtree,
          )
          plot.add(
            mark: "o",
            mark-size: .1,
            style: (mark: (stroke: red)),
            bg_nn,
          )
          plot.add(
            mark: "o",
            mark-size: .1,
            style: (mark: (stroke: green)),
            bg_knn,
          )
        })

        let legend = (
          "k-NN": (color: black, fill: green),
          "Neural Network": (color: black, fill: red),
          "Decision Tree": (color: black, fill: blue),
        );

        let x = 3
        let y = 0.8
        for (key, style) in legend {
          legend-item((x, y), key, style)
          y = y + 0.5
        }
    }),
    caption: [Plot of ensemble (bagging) with increasing bag size]
  )
]

From the table and figure above we can see a general trend for all three classifiers that as the bag size increases, the F1-Measure also increases. Hence the best performing ensemble size is the largest bag size, $20$. This phenomenon can be explained by the _Condorcet Jury Theorem_ where given the probability of each voter being correct is $p$ and the probability of majority of voters being correct is $M$, then: 

$ "if" p > 0.5, "then" M > p $ 
$ "if " p "always" > 0.5, "then" M "approaches" 1.0 "as the number of voters approaches" infinity $ 

In our case, as the bag size approaches infinity, the F1-measure of the ensemble approaches $1.0$. However, the computation cost of the ensemble also increases linearly with the bag size. Hence, there is a trade-off between the computation cost and the performance of the ensemble so the advisable approach is to keep increasing the bag size until improvements are too tiny to be considered or running out of computation power. 

Additionally, we can observe that for each classifier there seems to be a threshold bag size where increment after that have small impacts on the result. This is a phenomenon called "level-off" and will be discussed in the next section where the effect is more observable.

= Ensemble with random subspacing

== Results with increasing subspace size

The configuration for the results are as follows:
#list(
  indent: 1em,
  [*Decision Tree* $arrow.r$ `weka.classifiers.trees.J48 (10-fold cross validation`)],
  [*Neural Network* $arrow.r$ `weka.classifiers.functions.MultiLayerPerceptron (80/20 train/test split`)],
  [*k-NN* $arrow.r$ `weka.classifiers.lazy.IBk (10-fold cross validation)`]
)

The results below were ran using `weka.classifiers.meta.RandomSubSpace` with the above configuration and increasing `subSpaceSize` from $2$ to $20$. Note that, neural network does not use 10-fold cross validation because it takes too long to run. For the sake of simplicity, only the results for _Weighted Average_ is included.

#let rss_dtree = (
  (2, 0.482),
  (4, 0.527),
  (6, 0.556),
  (8, 0.572),
  (10, 0.573),
  (12, 0.522),
  (14, 0.522),
  (16, 0.522),
  (18, 0.522),
  (20, 0.522)
)
#let rss_nn = (
  (2, 0.451),
  (4, 0.507),
  (6, 0.533),
  (8, 0.559),
  (10, 0.558),
  (12, 0.557),
  (14, 0.557),
  (16, 0.557),
  (18, 0.567),
  (20, 0.567)
)
#let rss_knn = (
  (2, 0.415),
  (4, 0.480),
  (6, 0.517),
  (8, 0.527),
  (10, 0.514),
  (12, 0.488),
  (14, 0.488),
  (16, 0.488),
  (18, 0.488),
  (20, 0.488)
)

#align(center)[
  #figure(
    table(
      columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
      inset: 6pt,
      align: horizon,
      fill: (col, row) => if row == 0 or col == 0 { silver } else { white },
      [], [*2*], [*4*], [*6*], [*8*], [*10*], [*12*], [*14*], [*16*], [*18*], [*20*],
      [*Decision Tree (J48)*], ..rss_dtree.map(it => [#it.at(1)]),
      [*Neural \ Network*], ..rss_nn.map(it => [#it.at(1)]),
      [*k-NN*], ..rss_knn.map(it => [#it.at(1)])
    ),
    caption: [F1-Measure of the ensemble (random subspacing) with increasing sub space size]
  )
]

#align(center)[
  #figure(
    canvas(length: 1cm, {
      plot.plot(
        size: (8, 6),
        x-tick-step: 2,
        x-label: "Subspace Size",
        y-min: 0.4,
        y-max: 0.6,
        y-tick-step: 0.1,
        y-label: "Weighted F1-Measure",
        {
          plot.add(
            mark: "o",
            mark-size: .1,
            style: (mark: (stroke: blue)),
            rss_dtree,
          )
          plot.add(
            mark: "o",
            mark-size: .1,
            style: (mark: (stroke: red)),
            rss_nn,
          )
          plot.add(
            mark: "o",
            mark-size: .1,
            style: (mark: (stroke: green)),
            rss_knn,
          )
        })

        let legend = (
          "k-NN": (color: black, fill: green),
          "Neural Network": (color: black, fill: red),
          "Decision Tree": (color: black, fill: blue),
        );

        let x = 3
        let y = 0.8
        for (key, style) in legend {
          legend-item((x, y), key, style)
          y = y + 0.5
        }
    }),
    caption: [Plot of ensemble (random subspacing) with increasing sub space size]
  )
]

From the table and figure above, we see that from subspace size *$2$* to *$8$* all three classifiers shown an upward trend. However, after subspace size *$8$* the F1-Measure of all three classifiers started to level off and eventually plateus. This phenomenon is due to new ensemble members start to produce results similar to previous members hence no additional diversity are added causing the final output to be similar if not exact with the previous runs.

= Suitable classfiers each ensemble method

== Suitable classifiers for bagging ensemble

Based on the lectures, bagging is suitable for classifiers that are unstable. Unstable classifiers are classifiers that are sensitive to small change in the input data or in other words high variance. This is a sign that these classifiers are more prone to overfitting. Hence, bagging is suitable for unstable classifiers because it builds new models using the same classifier on variants of the data and if the classifier is very stable, it will just get similar results each time therefore not gaining much from the classifier. Several examples of unstable classifiers are decision trees and neural networks.

== Suitable classifiers for random subspacing ensemble

On the other hand, random subspacing is suitable for classifiers that are stable. Stable classifiers are classifiers that are not sensitive to small change in the input data or in other words low variance. Hence, random subspacing is suitable for stable classifiers because it introduced randomness by randomly selecting a subset of features each time and by doing so made the base models more diverse and hence reducing model correlation. One such stable classifier are k-NN.

== Best ensemble method for each classifer in this dataset

=== Decision Tree

#align(center)[
  #figure(
    canvas(length: 1cm, {
      plot.plot(
        size: (8, 6),
        x-tick-step: 2,
        x-label: "Ensemble Size",
        y-min: 0.4,
        y-max: 0.6,
        y-tick-step: 0.1,
        y-label: "Weighted F1-Measure",
        {
          plot.add(
            mark: "o",
            mark-size: .1,
            style: (mark: (stroke: blue)),
            bg_dtree,
          )
          plot.add(
            mark: "o",
            mark-size: .1,
            style: (mark: (stroke: red)),
            rss_dtree,
          )
        })

        let legend = (
          "Random Subspacing": (color: black, fill: red),
          "Bagging": (color: black, fill: blue),
        );

        let x = 2.4
        let y = 0.8
        for (key, style) in legend {
          legend-item((x, y), key, style)
          y = y + 0.5
        }
    }),
    caption: [Performance of bagging ensemble against random subspacing ensemble]
  )
]

The figure above shows the performance of bagging ensemble against random subspacing ensemble for decision tree. As can be seen from the figure, bagging ensemble performed better than random subspacing ensemble. This is in line with expectation because decision tree is an unstable classifier and hence bagging ensemble is more suitable for it.

=== Neural Network

#align(center)[
  #figure(
    canvas(length: 1cm, {
      plot.plot(
        size: (8, 6),
        x-tick-step: 2,
        x-label: "Ensemble Size",
        y-min: 0.4,
        y-max: 0.6,
        y-tick-step: 0.1,
        y-label: "Weighted F1-Measure",
        {
          plot.add(
            mark: "o",
            mark-size: .1,
            style: (mark: (stroke: blue)),
            bg_nn,
          )
          plot.add(
            mark: "o",
            mark-size: .1,
            style: (mark: (stroke: red)),
            rss_nn,
          )
        })

        let legend = (
          "Random Subspacing": (color: black, fill: red),
          "Bagging": (color: black, fill: blue),
        );

        let x = 2.4
        let y = 0.8
        for (key, style) in legend {
          legend-item((x, y), key, style)
          y = y + 0.5
        }
    }),
    caption: [Performance of bagging ensemble against random subspacing ensemble]
  )
]

From the comparison figure above, we can see that bagging ensemble performed better than random subspacing ensemble. This is in line with expectation because neural network is an unstable classifier and bagging ensemble takes advantage of this characteristics.

=== k-NN

#align(center)[
  #figure(
    canvas(length: 1cm, {
      plot.plot(
        size: (8, 6),
        x-tick-step: 2,
        x-label: "Ensemble Size",
        y-min: 0.4,
        y-max: 0.6,
        y-tick-step: 0.1,
        y-label: "Weighted F1-Measure",
        {
          plot.add(
            mark: "o",
            mark-size: .1,
            style: (mark: (stroke: blue)),
            bg_knn,
          )
          plot.add(
            mark: "o",
            mark-size: .1,
            style: (mark: (stroke: red)),
            rss_knn,
          )
        })

        let legend = (
          "Random Subspacing": (color: black, fill: red),
          "Bagging": (color: black, fill: blue),
        );

        let x = 2.4
        let y = 5.1
        for (key, style) in legend {
          legend-item((x, y), key, style)
          y = y + 0.5
        }
    }),
    caption: [Performance of bagging ensemble against random subspacing ensemble]
  )
]

From the figure above, the performance for both strategy seems similar but if we average each measure, we would get *#{calc.round(bg_knn.fold(0, (acc, it) => acc + it.at(1)) / bg_knn.len(), digits: 4)}* for bagging and *#{calc.round(rss_knn.fold(0, (acc, it) => acc + it.at(1)) / rss_knn.len(), digits: 4)}* for random subspacing. Hence, on average bagging ensemble performed better than random subspacing ensemble although the gain is almost negligible. Interestingly, random subspace performed better than bagging for ensemble size from *$6$* to *$10$*. This is a little out of expectation since k-NN is a stable classifier, I would expect random subspace to outperform bagging. One possible cause might be because I am using $k=1$ which might be too sensitive to noise and hence the ensemble is not able to mitigate the noise.

= Linear Regression and Stochastic Gradient Descent 

== Results for Linear Regression

The source code for the linear regression can be found at `src/LinearRegression.java`. To run the code, use the following command:

```sh
make run-linear-regression
```

This will compile the code and run linear regression on the dataset. The model trained with 80% of the data and evaluated with 10-fold cross validation using 20% of the data, results are as follows:

#align(center)[
  #figure(
    table(
      columns: (auto, auto),
      inset: 6pt,
      align: horizon,
      fill: (col, row) => if row == 0 or col == 0 { silver } else { white },
      [*Metrics*], [*Value*], 
      [*Correlation coefficient*], [0.6922], 
      [*Mean absolute error*], [0.0986],
      [*Root mean squared error*], [0.1248],
      [*Relative absolute error*], [71.2496%],
      [*Root relative squared error*], [72.1565%],
    ),
    caption: [Evaluation metrics for linear regression]
  )
]

Looking at the result, it seems that there indeed is a linear relationship between the features `tempo`, `loudness` and `liveness` and the target variable `energy` as proven by the correlation coefficient of *0.6922* indicating it has a positive correlation. However, the model is not doing very well since the relative absolute error (RAE) is *71.2496%* and the root relative squared error (RRSE) is *72.1565%*. This is probably due to the fact that the dataset is small and hence the model is not able to learn the relationship between the features and the target variable well enough.

== Results for Stochastic Gradient Descent

The source code for the linear regression can be found at `src/StochasticGradientDescent.java`. To run the code, use the following command:

```sh
make run-sgd
```

This will compile the code and run stochastic gradient descent with Squared Loss as the loss function on the dataset. The model trained with 80% of the data and evaluated with 10-fold cross validation using 20% of the data, results are as follows:

#align(center)[
  #figure(
    table(
      columns: (auto, auto),
      inset: 6pt,
      align: horizon,
      fill: (col, row) => if row == 0 or col == 0 { silver } else { white },
      [*Metrics*], [*Value*], 
      [*Correlation coefficient*], [0.6876], 
      [*Mean absolute error*], [0.0992],
      [*Root mean squared error*], [0.1256],
      [*Relative absolute error*], [71.6696%],
      [*Root relative squared error*], [72.6127%],
    ),
    caption: [Evaluation metrics for stochastic gradient descent]
  )
]

Surprisingly, the result for SGD is very similar to linear regression with its RAE and RRSE also in the 71% - 73% range. I would also consider that this model is also of poor quality because having a RAE of 71% means that the model is only able to predict the target variable correctly 29% of the time.

== Differences between Linear Regression and Stochastic Gradient Descent

From the results above, we can see that the performance for both linear regression and stochastic gradient descent (SGD) is very similar with subtle differences. There are several possible reasons for this:

#list(
  indent: 1em,
  [*Similar Loss Function* - The loss function that the SGD model uses is Squared Loss and the linear regression by Weka might also be using a similar underlying loss function.],
  [*Dataset Size* - The dataset size is small and therefore both models might not be able to learn the relationship between the features and the target variable well enough to make a difference.],
  [*Linearity* - The linear relationship between the features and target variable might be too dominant and hence SGD is basically optimising the same problem as linear regression on a similar gradient line. One evidence to support this is that the correlation coefficient for both models are relatively moderate and similar to each other.]
)
