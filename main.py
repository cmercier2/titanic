"""

"""

from titanic.data import load_data,clean_data,prepare_data
from titanic.registry import save_model
from titanic.train import train_model, evaluate_model, optimize_model


def main():
    df_train = clean_data(load_data("train"))
    df_test = clean_data(load_data("test"))
    X_train, y_train = prepare_data(df_train)
    X_test, y_test = prepare_data(df_test)

    lin = train_model(X_train, y_train)
    evaluate_model(X_test, y_test, lin)

    best_lin = optimize_model(X_train, y_train)
    evaluate_model(X_test, y_test, best_lin)

    save_model(best_lin)



if __name__=="__main__":
    main()