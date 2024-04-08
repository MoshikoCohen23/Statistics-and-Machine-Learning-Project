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

    # Example data for two columns (replace with your actual data)
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

    # Add legend
    plt.legend()

    # Set labels and title
    plt.xlabel('Grade')
    plt.ylabel('Density')
    plt.title('Probability Density Function of Grades for 1st and 2nd Semester')

    # Set x-axis limits
    plt.xlim(xmin, xmax)

    # Show the plot
    plt.show()

# t_test()

def outliers_samples():
    # Load your Excel data into a Pandas DataFrame
    excel_file_path = 'en_lpor_classification.csv'
    df = pd.read_csv(excel_file_path)
    df.dropna(inplace=True)

    # Selecting only the "Age" and "School Absence" columns
    selected_columns = ['Age', 'School Absence']
    selected_df = df[selected_columns]

    # Create box plots for each variable
    plt.figure(figsize=(10, 6))

    # Box plot for Age
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

# outliers_samples()

def graphs_of_explainatory_variables():
    # Assuming you have a DataFrame named 'df' with a column 'Category' containing the categories
    # Here's an example DataFrame for illustration purposes:
    data = pd.read_csv("en_lpor_explorer.csv")
    df = pd.DataFrame(data)
    df["Mother Education"] = df["Mother Education"].fillna("Without Education")
    df["Father Education"] = df["Father Education"].fillna("Without Education")

    ##school pie chart
    # Count the number of rows for each category
    category_counts = df['School'].value_counts()

    # Data for the pie chart
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
    # Count the number of rows for each category
    category_counts = df['Gender'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Gender Pie Chart")
    # Show the plot
    plt.show()

    ##age histogram
    # Create a histogram with a smoother KDE line
    sns.histplot(df['Age'], bins=20, color='skyblue', edgecolor='black', stat='density')

    # Set custom x-axis limits
    plt.xlim(0, 25)  # Adjust the limits according to your needs

    # Add labels and title
    plt.xlabel('Ages')
    plt.ylabel('Density')
    plt.title('Age Histogram')

    # Show the plot
    plt.show()

    # Get a summary of the histogram
    summary = df['Age'].describe()
    print(summary)

    ##address pie chart
    # Count the number of rows for each category
    category_counts = df['Housing Type'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Housing Type Pie Chart")
    # Show the plot
    plt.show()

    ##family size pie chart
    # Count the number of rows for each category
    category_counts = df['Family Size'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Family Size Pie Chart")
    # Show the plot
    plt.show()

    ##Parental_Status pie chart
    # Count the number of rows for each category
    category_counts = df['Parental Status'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Parental Status Pie Chart")
    # Show the plot
    plt.show()

    ##Mother_Education pie chart
    # Count the number of rows for each category
    category_counts = df['Mother Education'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Mother Education Pie Chart")
    # Show the plot
    plt.show()

    ##Father_Education pie chart
    # Count the number of rows for each category
    category_counts = df['Father Education'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Father Education Pie Chart")
    # Show the plot
    plt.show()

    ##Mother_Work pie chart
    # Count the number of rows for each category
    category_counts = df['Mother Work'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Mother Work Pie Chart")
    # Show the plot
    plt.show()

    ##Father_Work pie chart
    # Count the number of rows for each category
    category_counts = df['Father Work'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Father Work Pie Chart")
    # Show the plot
    plt.show()

    ##Reason_School_Choice pie chart
    # Count the number of rows for each category
    category_counts = df['Reason School Choice'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightgreen', 'lightsalmon']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Reason School Choice Pie Chart")
    # Show the plot
    plt.show()

    ##Legal_Responsibility pie chart
    # Count the number of rows for each category
    category_counts = df['Legal Responsibility'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightgreen']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Legal Responsibility Pie Chart")
    # Show the plot
    plt.show()

    ##Commute_Time pie chart
    # Count the number of rows for each category
    category_counts = df['Commute Time'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightgreen', 'lightsalmon']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Commute Time Pie Chart")
    # Show the plot
    plt.show()

    ##Weekly_Study_Time pie chart
    # Count the number of rows for each category
    category_counts = df['Weekly Study Time'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightgreen', 'lightsalmon']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Weekly Study Time Pie Chart")
    # Show the plot
    plt.show()

    ##Extra_Educational_Support pie chart
    # Count the number of rows for each category
    category_counts = df['Extra Educational Support'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Extra Educational Support Pie Chart")
    # Show the plot
    plt.show()

    ##Parental_Educational_Support pie chart
    # Count the number of rows for each category
    category_counts = df['Parental Educational Support'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Parental Educational Support Pie Chart")
    # Show the plot
    plt.show()

    ##Private_Tutoring pie chart
    # Count the number of rows for each category
    category_counts = df['Private Tutoring'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Private Tutoring Pie Chart")
    # Show the plot
    plt.show()

    ##Extracurricular_Activities pie chart
    # Count the number of rows for each category
    category_counts = df['Extracurricular Activities'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Extracurricular Activities Pie Chart")
    # Show the plot
    plt.show()

    ##Attended_Daycare pie chart
    # Count the number of rows for each category
    category_counts = df['Attended Daycare'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Attended Daycare Pie Chart")
    # Show the plot
    plt.show()

    ##Desire_Graduate_Education pie chart
    # Count the number of rows for each category
    category_counts = df['Desire Graduate Education'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Desire Graduate Education Pie Chart")
    # Show the plot
    plt.show()

    ##Has_Internet pie chart
    # Count the number of rows for each category
    category_counts = df['Has Internet'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Has Internet Pie Chart")
    # Show the plot
    plt.show()

    ##Is_Dating pie chart
    # Count the number of rows for each category
    category_counts = df['Is Dating'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Is Dating Pie Chart")
    # Show the plot
    plt.show()

    ##Good_Family_Relationship pie chart
    # Count the number of rows for each category
    category_counts = df['Good Family Relationship'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Good Family Relationship Pie Chart")
    # Show the plot
    plt.show()

    ##Free_Time_After_School pie chart
    # Count the number of rows for each category
    category_counts = df['Free Time After School'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Free Time After School Pie Chart")
    # Show the plot
    plt.show()

    ##Time_with_Friends pie chart
    # Count the number of rows for each category
    category_counts = df['Time with Friends'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Time with Friends Pie Chart")
    # Show the plot
    plt.show()

    ##Alcohol_Weekdays pie chart
    # Count the number of rows for each category
    category_counts = df['Alcohol Weekdays'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.0f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Alcohol Weekdays Pie Chart")
    # Show the plot
    plt.show()

    ##Alcohol_Weekends pie chart
    # Count the number of rows for each category
    category_counts = df['Alcohol Weekends'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Alcohol Weekends Pie Chart")
    # Show the plot
    plt.show()

    ##Health_Status pie chart
    # Count the number of rows for each category
    category_counts = df['Health Status'].value_counts()

    # Data for the pie chart
    labels = category_counts.index
    sizes = category_counts.values

    custom_colors = ['violet', 'skyblue', 'lightcoral','lightgreen', 'lightsalmon']

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.3f%%', startangle=90, colors = custom_colors)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title("Health Status Pie Chart")
    # Show the plot
    plt.show()

    ##School_Absence histogram
    # Create a histogram with a smoother KDE line
    sns.histplot(df['School Absence'], bins=94, color='skyblue', edgecolor='black', stat='density')

    # Set custom x-axis limits
    plt.xlim(0, 25)  # Adjust the limits according to your needs

    # Add labels and title
    plt.xlabel('Absence')
    plt.ylabel('Density')
    plt.title('School Absence Histogram')

    # Show the plot
    plt.show()


def feature_selection():
    # Load your Excel data into a Pandas DataFrame
    excel_file_path = 'en_lpor_explorer.csv'
    df_categories_with_names = pd.read_csv(excel_file_path)

    another_file = 'en_lpor_classification.csv'
    df_categories_with_numbers = pd.read_csv(another_file)

    df_categories_with_names["Mother Education"] = df_categories_with_names["Mother Education"].fillna("Without Education")
    df_categories_with_names["Father Education"] = df_categories_with_names["Father Education"].fillna("Without Education")

    # df_categories_with_numbers['Grade'] = np.where((df_categories_with_numbers['Grade'] >= 0) & (df_categories_with_numbers['Grade'] < 10), 0, 1)
    print(df_categories_with_numbers)
    print(df_categories_with_names)

    column_names = df_categories_with_numbers.columns.tolist()

    for name in column_names:
        if(name != "School Absence" or name != "Age"):
            df_categories_with_numbers[name] = df_categories_with_numbers[name].astype("category")


    # Separate features (X) and target variable (y)
    columns_to_drop = ['Grade 1st Semester', 'Grade 2nd Semester', 'Grade']
    X = df_categories_with_numbers.drop(columns=columns_to_drop, axis=1)
    y = df_categories_with_numbers['Grade']

    k = 12
    # Initialize the SelectKBest object with the f_classif scoring function
    selector = SelectKBest(score_func=f_regression, k=k)

    print(y)
    # Fit the selector to your data and transform it to select the top k features
    X_selected = selector.fit_transform(X, y)

    # Get the selected feature indices
    selected_feature_indices = selector.get_support(indices=True)

    # Get the names of the selected features
    selected_feature_names = X.columns[selected_feature_indices]

    # Get the scores of the selected features
    selected_feature_scores = selector.scores_[selected_feature_indices]

    # Create a DataFrame with selected feature names and scores
    selected_features_df = pd.DataFrame({'Feature': selected_feature_names, 'Score': selected_feature_scores})

    # Sort the DataFrame by score in descending order
    selected_features_df = selected_features_df.sort_values(by='Score', ascending=False)

    print(selected_features_df)
    # # Create dummy variables for selected features
    # X_dummies = pd.get_dummies(X[selected_feature_names], drop_first=True)
    #
    # # Combine dummy variables with the original DataFrame
    # X_combined = pd.concat([X, X_dummies], axis=1)

    # Selected Features and Scores Plot
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



    # Check column names in df1
    print("Column names in df1:", df_categories_with_names.columns)

    for feature_name in selected_feature_names:
    # # Get unique values from X_dummies, including 0
        unique_values = X[feature_name].unique()

        # Print column names in df1 for identification
        print("Column names in df1:", df_categories_with_names.columns)

        # Identify the correct column name in df1 using case-insensitive matching
        df1_column_name = next((col for col in df_categories_with_names.columns if col.lower() == feature_name.lower()), None)
        print(feature_name)
        if df1_column_name is not None:
            # Create a mapping based on the values in df1
            mapping_dict = dict(zip(df_categories_with_numbers[df1_column_name], df_categories_with_names[feature_name]))

            # Convert float keys to the correct type
            mapping_dict = {float(key): value for key, value in mapping_dict.items()}

            # Replace the original values with mapped values in X_dummies
            X_dummies_mapped = X.replace({feature_name: mapping_dict})

            # Create a new DataFrame with 'Grade' and the mapped feature
            df_mapped = pd.concat([df_categories_with_numbers['Grade'], X_dummies_mapped[feature_name]], axis=1)

            # Check unique values in the original and mapped columns
            print("Original Values:", X[feature_name].unique())
            print("Mapped Values:", df_mapped[feature_name].unique())

            # Now you can use df_mapped in the boxplot
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
            # Create a bar chart
            plt.figure(figsize=(8, 6))
            ax = df_mapped[feature_name].value_counts().sort_index().plot(kind='bar', color='skyblue')
            # Annotate each bar with its count
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

# feature_selection()

def unification_of_catefories():
    excel_file_path = 'en_lpor_classification.csv'
    df_categories_with_numbers = pd.read_csv(excel_file_path)

    excel_file_path = 'en_lpor_explorer.csv'
    df_categories_with_names = pd.read_csv(excel_file_path)


    column_names = df_categories_with_numbers.columns.tolist()

    for name in column_names:
        if (name != "School Absence" or name != "Grade"):
            df_categories_with_numbers[name] = df_categories_with_numbers[name].astype("category")

    # Separate features (X) and target variable (y)
    columns_to_drop = ['Grade 1st Semester', 'Grade 2nd Semester', 'Grade']
    X = df_categories_with_numbers.drop(columns=columns_to_drop, axis=1)
    y = df_categories_with_numbers['Grade']

    k = 12
    # Initialize the SelectKBest object with the f_classif scoring function
    selector = SelectKBest(score_func=f_regression, k=k)

    print(y)
    # Fit the selector to your data and transform it to select the top k features
    X_selected = selector.fit_transform(X, y)

    # Get the selected feature indices
    selected_feature_indices = selector.get_support(indices=True)

    # Get the names of the selected features
    selected_feature_names = X.columns[selected_feature_indices]

    # Get the scores of the selected features
    selected_feature_scores = selector.scores_[selected_feature_indices]

    # Create a DataFrame with selected feature names and scores
    selected_features_df = pd.DataFrame({'Feature': selected_feature_names, 'Score': selected_feature_scores})

    # Sort the DataFrame by score in descending order
    selected_features_df = selected_features_df.sort_values(by='Score', ascending=False)
    df = df_categories_with_names.filter(items=selected_feature_names)
    df["Grade"] = df_categories_with_names["Grade"]

    ### Mothers Education
    df['Mother Education'].fillna('Without education')
    # Define the mapping dictionary
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
    # Define the mapping dictionary
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
    # Define the mapping dictionary
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
    # Define the mapping dictionary
    Reason_School_Choice_mapping = {
        'Near Home': 'Near Home and Other',
        'Reputation': 'Reputation',
        'Course Preference': 'Course Preference',
        'Other': 'Near Home and Other'
    }
    # Replace values in the DataFrame
    df['Reason School Choice'] = df['Reason School Choice'].replace(Reason_School_Choice_mapping)


    ### Commute Time
    # Define the mapping dictionary
    Commute_Time_mapping = {
        'Up to 15 min': 'Up to 30 min',
        '15 to 30 min': 'Up to 30 min',
        '30 min to 1h': 'More than 30 min',
        'More than 1h': 'More than 30 min'
    }
    # Replace values in the DataFrame
    df['Commute Time'] = df['Commute Time'].replace(Commute_Time_mapping)


    ### Weekly Study Time
    # Define the mapping dictionary
    Weekly_Study_Time_mapping = {
        'Up to 2h': 'Up to 2h',
        '2 to 5h': 'More than 2h',
        '5 to 10h': 'More than 2h',
        'More than 10h': 'More than 2h'
    }
    # Replace values in the DataFrame
    df['Weekly Study Time'] = df['Weekly Study Time'].replace(Weekly_Study_Time_mapping)


    ### Alcohol Weekdays
    # Define the mapping dictionary
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
    # Define the mapping dictionary
    Alcohol_Weekends_mapping = {
        'Very Low': 'Low',
        'Low': 'Low',
        'Moderate': 'Low',
        'High': 'High',
        'Very High': 'High'
    }
    # Replace values in the DataFrame
    df['Alcohol Weekends'] = df['Alcohol Weekends'].replace(Alcohol_Weekends_mapping)

    # df['Grade'] = np.where((df['Grade'] >= 0) & (df['Grade'] < 10), 0, 1)
    # Specify the file path for the CSV file
    csv_file_path = 'regression.csv'

    # Write the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

# unification_of_catefories()
def feature_selection_logistic_regression():
    # Load your Excel data into a Pandas DataFrame
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


    # Separate features (X) and target variable (y)
    columns_to_drop = ['Grade 1st Semester', 'Grade 2nd Semester', 'Grade']
    X = df_categories_with_numbers.drop(columns=columns_to_drop, axis=1)
    y = df_categories_with_numbers['Grade']

    k = 12
    # Initialize the SelectKBest object with the f_classif scoring function
    selector = SelectKBest(score_func=f_regression, k=k)

    print(y)
    # Fit the selector to your data and transform it to select the top k features
    X_selected = selector.fit_transform(X, y)

    # Get the selected feature indices
    selected_feature_indices = selector.get_support(indices=True)

    # Get the names of the selected features
    selected_feature_names = X.columns[selected_feature_indices]

    # Get the scores of the selected features
    selected_feature_scores = selector.scores_[selected_feature_indices]

    # Create a DataFrame with selected feature names and scores
    selected_features_df = pd.DataFrame({'Feature': selected_feature_names, 'Score': selected_feature_scores})

    # Sort the DataFrame by score in descending order
    selected_features_df = selected_features_df.sort_values(by='Score', ascending=False)

    print(selected_features_df)
    # # Create dummy variables for selected features
    # X_dummies = pd.get_dummies(X[selected_feature_names], drop_first=True)
    #
    # # Combine dummy variables with the original DataFrame
    # X_combined = pd.concat([X, X_dummies], axis=1)

    # Selected Features and Scores Plot
    plt.figure(figsize=(10, 6))
    plt.bar(selected_features_df['Feature'], selected_features_df['Score'], color='blue')
    plt.xlabel('Selected Features')
    plt.ylabel('F-Score')
    plt.title('F-Scores of Selected Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# feature_selection_logistic_regression()
def unification_of_catefories_logistic_recression():
    excel_file_path = 'en_lpor_explorer.csv'
    df_categories_with_names = pd.read_csv(excel_file_path)
    df_categories_with_names = df_categories_with_names[['Desire Graduate Education','Time with Friends','Weekly Study Time','School','Private Tutoring','Commute Time','Free Time After School','Mother Education','Father Education','Grade']]
    ### Mothers Education
    df_categories_with_names['Mother Education'].fillna('Without education')
    # Define the mapping dictionary
    education_mapping = {
        'Without education': 'Low Education',
        'Primary School': 'Low Education',
        'Lower Secondary School': 'Secondary Education',
        'High School': 'High Education',
        'Higher Education': 'High Education'
    }
    # Replace values in the DataFrame
    df_categories_with_names['Mother Education'] = df_categories_with_names['Mother Education'].replace(education_mapping)

    ### Fathers Education
    df_categories_with_names['Father Education'].fillna('Without education')
    # Define the mapping dictionary
    education_mapping = {
        'Without education': 'Low Education',
        'Primary School': 'Low Education',
        'Lower Secondary School': 'Low Education',
        'High School': 'High Education',
        'Higher Education': 'High Education'
    }
    # Replace values in the DataFrame
    df_categories_with_names['Father Education'] = df_categories_with_names['Father Education'].replace(education_mapping)

    df_categories_with_names['Grade'] = np.where((df_categories_with_names['Grade'] >= 0) & (df_categories_with_names['Grade'] < 10), "Fail", "Pass")
    # Specify the file path for the CSV file
    csv_file_path = 'logistic_regression_excel.csv'

    # Write the DataFrame to a CSV file
    df_categories_with_names.to_csv(csv_file_path, index=False)

# unification_of_catefories_logistic_recression()

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


    # Apply the function to create a new column based on the condition
    df['Grade_Class'] = df['Grade'].apply(lambda x: condition(x))
    df.drop(columns='Grade', inplace=True)

    # samples_number_class_1 = df['Grade_Class'].sum()
    # print("samples number class 1:", samples_number_class_1)

    # Convert non-numeric columns to categorical variables
    non_numeric_columns = df.select_dtypes(include=['object']).columns
    df_dummies = pd.get_dummies(df, columns=non_numeric_columns)

    # -------------------- Perform feature selection ---------------------#
    X = df_dummies.drop(columns=['Grade_Class'])
    y = df_dummies['Grade_Class']
    np.random.seed(42)

    # Perform feature selection
    selector = SelectKBest(score_func=mutual_info_classif, k=13)  # Select top 13 features
    X_selected = selector.fit_transform(X, y)

    # Get the selected feature indices and their scores
    selected_feature_indices = selector.get_support(indices=True)
    feature_scores = selector.scores_

    # Get the original column names
    original_column_names = X.columns

    # Zip the original column names with their corresponding scores
    selected_features_with_scores = zip(original_column_names[selected_feature_indices],
                                        feature_scores[selected_feature_indices])

    # Convert to list for easier handling
    selected_features_with_scores = list(selected_features_with_scores)

    # Sort the zipped list based on feature scores
    sorted_features_with_scores = sorted(selected_features_with_scores, key=lambda x: x[1], reverse=True)

    # Extract the sorted feature names and scores
    sorted_features = [feature[0] for feature in sorted_features_with_scores]
    sorted_scores = [score[1] for score in sorted_features_with_scores]

    # Print the sorted features and their scores
    print("Sorted features by importance:")
    for feature, score in zip(sorted_features, sorted_scores):
        print(f"{feature}: {score}")

    # Save the sorted features and scores to a file
    with open("selected_features.txt", "w") as file:
        for feature, score in zip(sorted_features, sorted_scores):
            file.write(f"{feature}: {score}\n")

    original_selected_feature = []

    for dummy_feature in sorted_features:
        # Split the dummy feature name to extract the original feature name
        if ('_' in dummy_feature):
            original_feature, category = dummy_feature.split('_', 1)
            # Check if the original feature already exists in the mapping dictionary
            if original_feature not in original_selected_feature:
                original_selected_feature.append(original_feature)

        else:
            original_selected_feature.append(dummy_feature)

    # ---------------- implementation Random Forest Classifier -----------------#
    # Split the data into training and testing sets
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

    # Initialize the RandomForestClassifier
    forest = RandomForestClassifier(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(forest, param_grid, cv=5, scoring='f1')

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Predict on the test set using the best model
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    f1 = f1_score(y_test, y_pred)
    print("F1:", f1)

    # Plot feature importances
    feature_importances = best_model.feature_importances_

    # Get indices of top 12 features
    top_12_indices = np.argsort(feature_importances)[::-1][:12]

    # Get corresponding feature names
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
    # Split the data into training and testing sets
    X = df[original_selected_feature]
    X = pd.get_dummies(X)
    y = df['Grade_Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    param_grid = {
        'C': [0.1, 1],  # Regularization parameter
        'gamma': [0.1, 1],  # Kernel coefficient
        'kernel': ['linear', 'poly']  # Kernel type
    }

    # Initialize the SVM classifier
    svm_classifier = SVC()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='f1')

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Predict on the test set using the best model
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    f1 = accuracy_score(y_test, y_pred)
    print("f1:", f1)

    if best_params['kernel'] == 'linear':
        coef = best_model.coef_.ravel()
        feature_importances = np.abs(coef)
        feature_names = X.columns.tolist()  # Assuming X is a DataFrame and contains the feature names

        # Sort features based on importances and select the top 12
        top_indices = np.argsort(feature_importances)[-12:]
        top_feature_importances = feature_importances[top_indices]
        top_feature_names = [feature_names[i] for i in top_indices]

        plt.figure(figsize=(10, 6))
        plt.barh(top_feature_names, top_feature_importances)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top 12 Feature Importance')
        plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
        plt.show()
    else:
        # For non-linear kernels, it's not straightforward to extract feature importances
        # You may need to use other techniques like permutation importance or SHAP values
        print("Feature importances not available for non-linear kernels")

    # ---------------- implementation Neural Network Claasifier (NNC) -----------------#
    # Split the data into training and testing sets
    X = df_dummies[sorted_features]
    y = df_dummies['Grade_Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (150,)],  # Number of neurons in hidden layers
        'alpha': [0.0001, 0.001, 0.01],  # Regularization parameter
        'learning_rate_init': [0.001, 0.01, 0.1]  # Initial learning rate
    }

    # Initialize the MLP classifier with early stopping
    mlp_classifier = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.1,
                                   n_iter_no_change=10)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(mlp_classifier, param_grid, cv=5, scoring='f1')

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Predict on the test set using the best model
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    f1 = accuracy_score(y_test, y_pred)
    print("Accuracy:", f1)

    # Get the coefficients of the input layer (assuming first layer is the input layer)
    coefs_input_layer = best_model.coefs_[0]

    # Calculate the mean absolute value of coefficients across all input features
    feature_importances = np.mean(np.abs(coefs_input_layer), axis=1)

    # Sort feature importances and corresponding feature names
    sorted_indices = feature_importances.argsort()[::-1]
    sorted_feature_importances = feature_importances[sorted_indices]
    sorted_feature_names = np.array(sorted_features)[sorted_indices]

    # Create a bar plot of feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_feature_importances)), sorted_feature_importances, align='center')
    plt.yticks(range(len(sorted_feature_importances)), sorted_feature_names)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance of Input Features')
    plt.gca().invert_yaxis()
    plt.show()

# ML_models()