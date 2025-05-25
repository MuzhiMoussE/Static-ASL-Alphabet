1. Train new models or use the existing models:
Run the script without the --use_model argument or simply run the main.py:
    python main.py
2. Use Existing Model: Specify the model to use with --use_model:
    python main.py --use_model knn
    python main.py --use_model svm
    python main.py --use_model naive_bayes
You can also specify the parameters:
    python main.py --n_neighbor [Number of Neighbor] --c_svm [Value of in SVM]
If you want to train a new model, you need to delete the models in the `models` directory and run the script again.
3. Predict New Data**: To predict new data, use the `--predict` argument:
    python main.py --use_model [Model] --predict [Image Path]

-----Utils-----

1. To generate and save the SVM performance plot with different `C` values, use the following command:
    python main.py --plot_svm
    The plot will be saved in the `results` directory as `svm_performance.png`.
2. To generate and save the KNN performance plot with different `n_neighbors` values, use the following command:
    python main.py --plot_knn
    The plot will be saved in the `results` directory as `knn_performance.png`.
3. To generate and save the Naive Bayes performance plot, use the following command:
    python main.py --plot_naive_bayes
4. To generate and save the cross-validation performance plot, use the following command:
    python main.py --run_cv
5. To generate and save the learning curve plot, use the following command:
    python main.py --run_learning_curve
6. To generate and save the confusion matrix plot, use the following command:
    python main.py  --plot_confusion
