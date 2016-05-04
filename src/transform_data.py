import os
import pandas as pd

data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')

i_train = os.path.join(data_dir, 'raw', 'train.csv')
i_test = os.path.join(data_dir, 'raw', 'test.csv')

train_df = pd.read_csv(i_train)
test_df = pd.read_csv(i_test)

combo_df = pd.concat((train_df, test_df), axis=0, ignore_index=True)
fill_ages = combo_df[['Age', 'Sex', 'Pclass']].groupby(['Sex', 'Pclass']).median()


def age_fill(row):
    return fill_ages.loc[row.Sex].loc[row.Pclass]


for csv in [i_train, i_test]:
    df = pd.read_csv(csv)
    df = df.drop(['Ticket', 'Name', 'Embarked', 'Cabin'], axis=1)

    # fill in missing age data
    filler_ages = df[['Age', 'Sex', 'Pclass']].groupby(['Sex', 'Pclass']).median()

    mask_age = df['Age'].notnull()
    df_age = df.loc[mask_age, ['Age', 'Sex', 'Pclass']]

    mask_Age = df['Age'].isnull()
    df_age_miss = df.loc[mask_Age, ['Sex', 'Pclass']]

    df_age_miss['Age'] = df_age_miss.apply(age_fill, axis=1)

    df['Age'] = pd.concat([df_age['Age'], df_age_miss['Age']])

    # fill in remaining NaN values
    df = df.fillna(df.median())

    # set up dummy variables for passenger class
    # class_dummies = pd.get_dummies(df['Pclass'])
    # class_dummies = class_dummies.rename(columns={1: 'Pclass1', 2: 'Pclass2', 3: 'Pclass3'})
    # df = pd.concat([df, class_dummies], axis=1)

    # set up dummy variables for gender
    sex_dummies = pd.get_dummies(df['Sex'])
    df = pd.concat([df, sex_dummies], axis=1)

    drop_cols = ['Sex']

    if csv == i_train:
        df.drop(drop_cols, 1).to_csv(os.path.join(data_dir, 'interim', 'train.csv'), index=False)
    elif csv == i_test:
        df.drop(drop_cols, 1).to_csv(os.path.join(data_dir, 'interim', 'test.csv'), index=False)

    # feature engineering
    # df['AgeClass'] = df['Age'] * df['Pclass']
    # df['FamilySize'] = df['Parch'] + df['SibSp']

    if csv == i_train:
        df.drop(drop_cols, 1).to_csv(os.path.join(data_dir, 'processed', 'train.csv'), index=False)
    elif csv == i_test:
        df.drop(drop_cols, 1).to_csv(os.path.join(data_dir, 'processed', 'test.csv'), index=False)
