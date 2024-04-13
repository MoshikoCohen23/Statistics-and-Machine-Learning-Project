import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from scipy.stats import f_oneway
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def t_test():
    excel_file_path = 'en_lpor_classification.csv'
    df = pd.read_csv(excel_file_path)
    df.dropna(inplace=True)

    column1 = df['Grade 1st Semester']
    column2 = df['Grade 2nd Semester']

    # Perform paired t-test
    t_statistic, p_value = ttest_rel(column1, column2)

    # Output the results
    print("Paired t-test results:")
    print("t-statistic:", t_statistic)
    print("p-value:", p_value)

    # Interpret the p-value
    alpha = 0.05  # Significance level
    if p_value < alpha:
        print("Reject the null hypothesis. There is a significant difference between the two columns.")
    else:
        print("Fail to reject the null hypothesis. There is no significant difference between the two columns.")

    mean_1st_Semester = column1.mean()
    mean_2nd_Semester = column2.mean()

    data1 = np.random.normal(loc=10, scale=2, size=1000)  # Sample data for column 1
    data2 = np.random.normal(loc=12, scale=3, size=1000)  # Sample data for column 2

    # Fit a normal distribution to the data
    mu1, std1 = norm.fit(data1)
    mu2, std2 = norm.fit(data2)

    # Plot the histograms
    plt.hist(data1, bins=30, density=True, alpha=0.5, color='blue', label='Grade 1st Semester')
    plt.hist(data2, bins=30, density=True, alpha=0.5, color='red', label='Grade 2nd Semester')

    # Plot the PDF curves
    xmin, xmax = 0, 20  # Set the x-axis limits
    x = np.linspace(xmin, xmax, 100)

    p1 = norm.pdf(x, mu1, std1)
    plt.plot(x, p1, 'b', linewidth=2)

    p2 = norm.pdf(x, mu2, std2)
    plt.plot(x, p2, 'r', linewidth=2)

    plt.legend()

    plt.xlabel('Grade')
    plt.ylabel('Density')
    plt.title('Probability Density Function of Grades for 1st and 2nd Semester')

    plt.xlim(xmin, xmax)

    plt.show()


def outliers_samples():
    excel_file_path = 'en_lpor_classification.csv'
    df = pd.read_csv(excel_file_path)
    df.dropna(inplace=True)

    # Selecting only the "Age" and "School Absence" columns
    selected_columns = ['Age', 'School Absence']
    selected_df = df[selected_columns]

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.boxplot(selected_df['Age'])
    plt.title('Box Plot of Age')
    plt.ylabel('Age')
    plt.xlabel('')

    # Box plot for School Absence
    plt.subplot(1, 2, 2)
    plt.boxplot(selected_df['School Absence'])
    plt.title('Box Plot of School Absences')
    plt.ylabel('School Absences')
    plt.xlabel('')

    plt.tight_layout()
    plt.show()


def graphs_of_explainatory_variables():
    data = pd.read_csv("en_lpor_explorer.csv")
    df = pd.DataFrame(data)
    df["Mother Education"] = df["Mother Education"].fillna("Without Education")
    df["Father Education"] = df["Father Education"].fillna("Without Education")

    ##school pie chart
    # Count the number of rows for each category
    category_counts = df['School'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("School Pie Chart")
    # Show the plot
    plt.show()

    ##gender pie chart
    category_counts = df['Gender'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Gender Pie Chart")
    plt.show()

    ##age histogram
    sns.histplot(df['Age'], bins=20, color='skyblue', edgecolor='black', stat='density')
    plt.xlim(0, 25)  

    # Add labels and title
    plt.xlabel('Ages')
    plt.ylabel('Density')
    plt.title('Age Histogram')

    plt.show()

    summary = df['Age'].describe()
    print(summary)

    # Count the number of rows for each category
    category_counts = df['Housing Type'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Housing Type Pie Chart")
    plt.show()

    # Count the number of rows for each category
    category_counts = df['Family Size'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Family Size Pie Chart")
    plt.show()

    ##Parental_Status pie chart
    category_counts = df['Parental Status'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Parental Status Pie Chart")
    plt.show()

    ##Mother_Education pie chart
    category_counts = df['Mother Education'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Mother Education Pie Chart")
    plt.show()

    ##Father_Education pie chart
    category_counts = df['Father Education'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Father Education Pie Chart")
    plt.show()

    ##Mother_Work pie chart
    category_counts = df['Mother Work'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Mother Work Pie Chart")
    plt.show()

    ##Father_Work pie chart
    category_counts = df['Father Work'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Father Work Pie Chart")
    plt.show()

    ##Reason_School_Choice pie chart
    category_counts = df['Reason School Choice'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightgreen', 'lightsalmon']

    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Reason School Choice Pie Chart")
    plt.show()

    ##Legal_Responsibility pie chart
    category_counts = df['Legal Responsibility'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightgreen']

    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Legal Responsibility Pie Chart")
    plt.show()

    ##Commute_Time pie chart
    category_counts = df['Commute Time'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightgreen', 'lightsalmon']

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Commute Time Pie Chart")
    plt.show()

    ##Weekly_Study_Time pie chart
    category_counts = df['Weekly Study Time'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightgreen', 'lightsalmon']

    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Weekly Study Time Pie Chart")
    plt.show()

    ##Extra_Educational_Support pie chart
    category_counts = df['Extra Educational Support'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Extra Educational Support Pie Chart")
    plt.show()

    ##Parental_Educational_Support pie chart
    category_counts = df['Parental Educational Support'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Parental Educational Support Pie Chart")
    plt.show()

    ##Private_Tutoring pie chart
    category_counts = df['Private Tutoring'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Private Tutoring Pie Chart")
    plt.show()

    ##Extracurricular_Activities pie chart
    category_counts = df['Extracurricular Activities'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values
    custom_colors = ['violet', 'skyblue']

    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Extracurricular Activities Pie Chart")
    plt.show()

    ##Attended_Daycare pie chart
    category_counts = df['Attended Daycare'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Attended Daycare Pie Chart")
    plt.show()

    ##Desire_Graduate_Education pie chart
    category_counts = df['Desire Graduate Education'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Desire Graduate Education Pie Chart")
    plt.show()

    ##Has_Internet pie chart
    category_counts = df['Has Internet'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Has Internet Pie Chart")
    plt.show()

    ##Is_Dating pie chart
    category_counts = df['Is Dating'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Is Dating Pie Chart")
    plt.show()

    ##Good_Family_Relationship pie chart
    category_counts = df['Good Family Relationship'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Good Family Relationship Pie Chart")
    plt.show()

    ##Free_Time_After_School pie chart
    category_counts = df['Free Time After School'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Free Time After School Pie Chart")
    plt.show()

    ##Time_with_Friends pie chart
    category_counts = df['Time with Friends'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Time with Friends Pie Chart")
    plt.show()

    ##Alcohol_Weekdays pie chart
    category_counts = df['Alcohol Weekdays'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    plt.pie(sizes, labels=labels, autopct='%1.0f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Alcohol Weekdays Pie Chart")
    plt.show()

    ##Alcohol_Weekends pie chart
    category_counts = df['Alcohol Weekends'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Alcohol Weekends Pie Chart")
    plt.show()

    ##Health_Status pie chart
    category_counts = df['Health Status'].value_counts()

    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    plt.axis('equal')
    plt.title("Health Status Pie Chart")
    plt.show()

    ##School_Absence histogram
    sns.histplot(df['School Absence'], bins=94, color='skyblue', edgecolor='black', stat='density')

    plt.xlim(0, 25)  

    plt.xlabel('Absence')
    plt.ylabel('Density')
    plt.title('School Absence Histogram')

    plt.show()


def feature_selection():
    excel_file_path = 'en_lpor_explorer.csv'
    df_categories_with_names = pd.read_csv(excel_file_path)

    another_file = 'en_lpor_classification.csv'
    df_categories_with_numbers = pd.read_csv(another_file)

    df_categories_with_names["Mother Education"] = df_categories_with_names["Mother Education"].fillna("Without Education")
    df_categories_with_names["Father Education"] = df_categories_with_names["Father Education"].fillna("Without Education")

    print(df_categories_with_numbers)
    print(df_categories_with_names)

    column_names = df_categories_with_numbers.columns.tolist()

    for name in column_names:
        if(name != "School Absence" or name != "Age"):
            df_categories_with_numbers[name] = df_categories_with_numbers[name].astype("category")


    columns_to_drop = ['Grade 1st Semester', 'Grade 2nd Semester', 'Grade']
    X = df_categories_with_numbers.drop(columns=columns_to_drop, axis=1)
    y = df_categories_with_numbers['Grade']

    k = 12
    selector = SelectKBest(score_func=f_regression, k=k)

    print(y)
    X_selected = selector.fit_transform(X, y)

    selected_feature_indices = selector.get_support(indices=True)

    selected_feature_names = X.columns[selected_feature_indices]

    selected_feature_scores = selector.scores_[selected_feature_indices]

    selected_features_df = pd.DataFrame({'Feature': selected_feature_names, 'Score': selected_feature_scores})

    selected_features_df = selected_features_df.sort_values(by='Score', ascending=False)

    print(selected_features_df)
    plt.figure(figsize=(10, 6))
    plt.bar(selected_features_df['Feature'], selected_features_df['Score'], color='blue')
    plt.xlabel('Selected Features')
    plt.ylabel('F-Score')
    plt.title('F-Scores of Selected Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    ######### pairwise_tukeyhsd
    for feature in selected_features_df['Feature']:
        # Perform Tukey's HSD test
        print(feature)
        tukey_results = pairwise_tukeyhsd(df_categories_with_numbers['Grade'], df_categories_with_numbers[feature], alpha=0.05)

        # Print the summary of the results
        print(f"Tukey's HSD results for {feature}:")
        print(tukey_results.summary())


    print("Column names in df1:", df_categories_with_names.columns)

    for feature_name in selected_feature_names:
        unique_values = X[feature_name].unique()

        print("Column names in df1:", df_categories_with_names.columns)

        df1_column_name = next((col for col in df_categories_with_names.columns if col.lower() == feature_name.lower()), None)
        print(feature_name)
        if df1_column_name is not None:
            mapping_dict = dict(zip(df_categories_with_numbers[df1_column_name], df_categories_with_names[feature_name]))

            mapping_dict = {float(key): value for key, value in mapping_dict.items()}

            X_dummies_mapped = X.replace({feature_name: mapping_dict})

            df_mapped = pd.concat([df_categories_with_numbers['Grade'], X_dummies_mapped[feature_name]], axis=1)

            print("Original Values:", X[feature_name].unique())
            print("Mapped Values:", df_mapped[feature_name].unique())

            plt.figure(figsize=(12, 8))

            if len(df_mapped[feature_name].unique()) >= 2:
                ax = df_mapped.boxplot(column='Grade', by=feature_name, grid=False, patch_artist=True)
                plt.xlabel(f'{feature_name}')
                plt.ylabel('Grade')

                # Perform one-way ANOVA to get F-statistic and p-value
                groups = [df_mapped[df_mapped[feature_name] == group]['Grade'] for group in df_mapped[feature_name].unique()]
                f_statistic, p_value = f_oneway(*groups)

                # Add F-test information to the plot
                title_str = f' p-value of F test: {p_value:.4f}'  # Append p-value to the title
                plt.title(title_str)
                ax.set_title(title_str, fontsize=8)
            else:
                print("Error: There are not enough unique groups for one-way ANOVA.")

            plt.show()
            plt.figure(figsize=(8, 6))
            ax = df_mapped[feature_name].value_counts().sort_index().plot(kind='bar', color='skyblue')
            for p in ax.patches:
                ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 10), textcoords='offset points')

            plt.xticks(rotation=0)
            plt.title(f'Number of Samples of {feature_name} Category')
            plt.xlabel(f'{feature_name}')
            plt.ylabel('Count')
            plt.show()
        else:
            print(f"Error: Column '{feature_name}' not found in df1.")


def unification_of_catefories():
    excel_file_path = 'en_lpor_classification.csv'
    df_categories_with_numbers = pd.read_csv(excel_file_path)

    excel_file_path = 'en_lpor_explorer.csv'
    df_categories_with_names = pd.read_csv(excel_file_path)


    column_names = df_categories_with_numbers.columns.tolist()

    for name in column_names:
        if (name != "School Absence" or name != "Grade"):
            df_categories_with_numbers[name] = df_categories_with_numbers[name].astype("category")

    columns_to_drop = ['Grade 1st Semester', 'Grade 2nd Semester', 'Grade']
    X = df_categories_with_numbers.drop(columns=columns_to_drop, axis=1)
    y = df_categories_with_numbers['Grade']

    k = 12
    selector = SelectKBest(score_func=f_regression, k=k)

    print(y)
    X_selected = selector.fit_transform(X, y)

    selected_feature_indices = selector.get_support(indices=True)

    selected_feature_names = X.columns[selected_feature_indices]

    selected_feature_scores = selector.scores_[selected_feature_indices]

    selected_features_df = pd.DataFrame({'Feature': selected_feature_names, 'Score': selected_feature_scores})

    selected_features_df = selected_features_df.sort_values(by='Score', ascending=False)
    df = df_categories_with_names.filter(items=selected_feature_names)
    df["Grade"] = df_categories_with_names["Grade"]

    ### Mothers Education
    df['Mother Education'].fillna('Without education')
    education_mapping = {
        'Without education': 'Low Education',
        'Primary School': 'Low Education',
        'Lower Secondary School': 'Secondary Education',
        'High School': 'High Education',
        'Higher Education': 'High Education'
    }
    # Replace values in the DataFrame
    df['Mother Education'] = df['Mother Education'].replace(education_mapping)


    ### Fathers Education
    df['Father Education'].fillna('Without education')
    education_mapping = {
        'Without education': 'Low Education',
        'Primary School': 'Low Education',
        'Lower Secondary School': 'Low Education',
        'High School': 'High Education',
        'Higher Education': 'High Education'
    }
    # Replace values in the DataFrame
    df['Father Education'] = df['Father Education'].replace(education_mapping)


    ### Mothers Work
    work_mapping = {
        'Teacher': 'Work',
        'Health': 'Work',
        'Services': 'Work',
        'Homemaker': 'Homemaker',
        'other': 'Work'
    }
    # Replace values in the DataFrame
    df['Mother Work'] = df['Mother Work'].replace(work_mapping)


    ### Reason School Choice
    Reason_School_Choice_mapping = {
        'Near Home': 'Near Home and Other',
        'Reputation': 'Reputation',
        'Course Preference': 'Course Preference',
        'Other': 'Near Home and Other'
    }
    # Replace values in the DataFrame
    df['Reason School Choice'] = df['Reason School Choice'].replace(Reason_School_Choice_mapping)


    ### Commute Time
    Commute_Time_mapping = {
        'Up to 15 min': 'Up to 30 min',
        '15 to 30 min': 'Up to 30 min',
        '30 min to 1h': 'More than 30 min',
        'More than 1h': 'More than 30 min'
    }
    # Replace values in the DataFrame
    df['Commute Time'] = df['Commute Time'].replace(Commute_Time_mapping)


    ### Weekly Study Time
    Weekly_Study_Time_mapping = {
        'Up to 2h': 'Up to 2h',
        '2 to 5h': 'More than 2h',
        '5 to 10h': 'More than 2h',
        'More than 10h': 'More than 2h'
    }
    # Replace values in the DataFrame
    df['Weekly Study Time'] = df['Weekly Study Time'].replace(Weekly_Study_Time_mapping)


    ### Alcohol Weekdays
    Alcohol_Weekdays_mapping = {
        'Very Low': 'Low',
        'Low': 'Low',
        'Moderate': 'High',
        'High': 'High',
        'Very High': 'High'
    }
    # Replace values in the DataFrame
    df['Alcohol Weekdays'] = df['Alcohol Weekdays'].replace(Alcohol_Weekdays_mapping)


    ### Alcohol Weekends
    Alcohol_Weekends_mapping = {
        'Very Low': 'Low',
        'Low': 'Low',
        'Moderate': 'Low',
        'High': 'High',
        'Very High': 'High'
    }
    # Replace values in the DataFrame
    df['Alcohol Weekends'] = df['Alcohol Weekends'].replace(Alcohol_Weekends_mapping)

    csv_file_path = 'regression.csv'
    df.to_csv(csv_file_path, index=False)

def feature_selection_logistic_regression():
    excel_file_path = 'en_lpor_explorer.csv'
    df_categories_with_names = pd.read_csv(excel_file_path)

    another_file = 'en_lpor_classification.csv'
    df_categories_with_numbers = pd.read_csv(another_file)

    df_categories_with_names["Mother Education"] = df_categories_with_names["Mother Education"].fillna("Without Education")
    df_categories_with_names["Father Education"] = df_categories_with_names["Father Education"].fillna("Without Education")

    df_categories_with_numbers['Grade'] = np.where((df_categories_with_numbers['Grade'] >= 0) & (df_categories_with_numbers['Grade'] < 10), 0, 1)
    print(df_categories_with_numbers)
    print(df_categories_with_names)

    column_names = df_categories_with_numbers.columns.tolist()

    for name in column_names:
        if(name != "School Absence" or name != "Age"):
            df_categories_with_numbers[name] = df_categories_with_numbers[name].astype("category")


    columns_to_drop = ['Grade 1st Semester', 'Grade 2nd Semester', 'Grade']
    X = df_categories_with_numbers.drop(columns=columns_to_drop, axis=1)
    y = df_categories_with_numbers['Grade']

    k = 12
    selector = SelectKBest(score_func=f_regression, k=k)

    print(y)
    X_selected = selector.fit_transform(X, y)

    selected_feature_indices = selector.get_support(indices=True)
    selected_feature_names = X.columns[selected_feature_indices]
    selected_feature_scores = selector.scores_[selected_feature_indices]
    selected_features_df = pd.DataFrame({'Feature': selected_feature_names, 'Score': selected_feature_scores})
    selected_features_df = selected_features_df.sort_values(by='Score', ascending=False)
    print(selected_features_df)

    plt.figure(figsize=(10, 6))
    plt.bar(selected_features_df['Feature'], selected_features_df['Score'], color='blue')
    plt.xlabel('Selected Features')
    plt.ylabel('F-Score')
    plt.title('F-Scores of Selected Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def unification_of_catefories_logistic_recression():
    excel_file_path = 'en_lpor_explorer.csv'
    df_categories_with_names = pd.read_csv(excel_file_path)
    df_categories_with_names = df_categories_with_names[['Desire Graduate Education','Time with Friends','Weekly Study Time','School','Private Tutoring','Commute Time','Free Time After School','Mother Education','Father Education','Grade']]
    ### Mothers Education
    df_categories_with_names['Mother Education'].fillna('Without education')
    education_mapping = {
        'Without education': 'Low Education',
        'Primary School': 'Low Education',
        'Lower Secondary School': 'Secondary Education',
        'High School': 'High Education',
        'Higher Education': 'High Education'
    }
    df_categories_with_names['Mother Education'] = df_categories_with_names['Mother Education'].replace(education_mapping)

    ### Fathers Education
    df_categories_with_names['Father Education'].fillna('Without education')
    education_mapping = {
        'Without education': 'Low Education',
        'Primary School': 'Low Education',
        'Lower Secondary School': 'Low Education',
        'High School': 'High Education',
        'Higher Education': 'High Education'
    }
    df_categories_with_names['Father Education'] = df_categories_with_names['Father Education'].replace(education_mapping)
    df_categories_with_names['Grade'] = np.where((df_categories_with_names['Grade'] >= 0) & (df_categories_with_names['Grade'] < 10), "Fail", "Pass")
    csv_file_path = 'logistic_regression_excel.csv'
    df_categories_with_names.to_csv(csv_file_path, index=False)


def ML_models():

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    import matplotlib.pyplot as plt
    import numpy as np

    # ------------------------------------------- Classification --------------------------------------#
    excel_file_path = 'en_lpor_explorer.csv'
    df = pd.read_csv(excel_file_path)
    columns_to_drop = ['Grade 1st Semester', 'Grade 2nd Semester']
    df.drop(columns=columns_to_drop, inplace=True)


    def condition(x):
        if x >= 10:
            return 1
        else:
            return 0


    df['Grade_Class'] = df['Grade'].apply(lambda x: condition(x))
    df.drop(columns='Grade', inplace=True)

    non_numeric_columns = df.select_dtypes(include=['object']).columns
    df_dummies = pd.get_dummies(df, columns=non_numeric_columns)

    # -------------------- Perform feature selection ---------------------#
    X = df_dummies.drop(columns=['Grade_Class'])
    y = df_dummies['Grade_Class']
    np.random.seed(42)

    # Perform feature selection
    selector = SelectKBest(score_func=mutual_info_classif, k=13)  # Select top 13 features
    X_selected = selector.fit_transform(X, y)

    selected_feature_indices = selector.get_support(indices=True)
    feature_scores = selector.scores_
    original_column_names = X.columns

    selected_features_with_scores = zip(original_column_names[selected_feature_indices],
                                        feature_scores[selected_feature_indices])

    selected_features_with_scores = list(selected_features_with_scores)

    sorted_features_with_scores = sorted(selected_features_with_scores, key=lambda x: x[1], reverse=True)

    sorted_features = [feature[0] for feature in sorted_features_with_scores]
    sorted_scores = [score[1] for score in sorted_features_with_scores]

    print("Sorted features by importance:")
    for feature, score in zip(sorted_features, sorted_scores):
        print(f"{feature}: {score}")

    # Save the sorted features and scores to a file
    with open("selected_features.txt", "w") as file:
        for feature, score in zip(sorted_features, sorted_scores):
            file.write(f"{feature}: {score}\n")

    original_selected_feature = []

    for dummy_feature in sorted_features:
        if ('_' in dummy_feature):
            original_feature, category = dummy_feature.split('_', 1)
            if original_feature not in original_selected_feature:
                original_selected_feature.append(original_feature)

        else:
            original_selected_feature.append(dummy_feature)

    # ---------------- implementation Random Forest Classifier -----------------#
    X = df[original_selected_feature]
    X = pd.get_dummies(X)
    y = df['Grade_Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the parameter grid for grid search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [2, 4, 6]
    }

    forest = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(forest, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    print("F1:", f1)
    feature_importances = best_model.feature_importances_
    top_12_indices = np.argsort(feature_importances)[::-1][:12]
    top_12_features = X.columns[top_12_indices]
    top_12_importances = feature_importances[top_12_indices]

    # Plot top 12 feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(top_12_features)), top_12_importances, color='b', alpha=0.6)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Top 12 Feature Importances')
    plt.xticks(range(len(top_12_features)), top_12_features, rotation=45)
    plt.tight_layout()
    plt.show()

    # ---------------- implementation Support Vector Machine (SVM) -----------------#
    X = df[original_selected_feature]
    X = pd.get_dummies(X)
    y = df['Grade_Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    param_grid = {
        'C': [0.1, 1],  # Regularization parameter
        'gamma': [0.1, 1],  # Kernel coefficient
        'kernel': ['linear', 'poly']  # Kernel type
    }

    svm_classifier = SVC()
    grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    f1 = accuracy_score(y_test, y_pred)
    print("f1:", f1)

    if best_params['kernel'] == 'linear':
        coef = best_model.coef_.ravel()
        feature_importances = np.abs(coef)
        feature_names = X.columns.tolist()  

        top_indices = np.argsort(feature_importances)[-12:]
        top_feature_importances = feature_importances[top_indices]
        top_feature_names = [feature_names[i] for i in top_indices]

        plt.figure(figsize=(10, 6))
        plt.barh(top_feature_names, top_feature_importances)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top 12 Feature Importance')
        plt.gca().invert_yaxis() 
        plt.show()
    else:

        print("Feature importances not available for non-linear kernels")

    # ---------------- implementation Neural Network Claasifier (NNC) -----------------#
    X = df_dummies[sorted_features]
    y = df_dummies['Grade_Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (150,)],  # Number of neurons in hidden layers
        'alpha': [0.0001, 0.001, 0.01],  # Regularization parameter
        'learning_rate_init': [0.001, 0.01, 0.1]  # Initial learning rate
    }

    mlp_classifier = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.1,
                                   n_iter_no_change=10)
    grid_search = GridSearchCV(mlp_classifier, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    f1 = accuracy_score(y_test, y_pred)
    print("Accuracy:", f1)

    coefs_input_layer = best_model.coefs_[0]
    feature_importances = np.mean(np.abs(coefs_input_layer), axis=1)

    sorted_indices = feature_importances.argsort()[::-1]
    sorted_feature_importances = feature_importances[sorted_indices]
    sorted_feature_names = np.array(sorted_features)[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_feature_importances)), sorted_feature_importances, align='center')
    plt.yticks(range(len(sorted_feature_importances)), sorted_feature_names)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance of Input Features')
    plt.gca().invert_yaxis()
    plt.show()

