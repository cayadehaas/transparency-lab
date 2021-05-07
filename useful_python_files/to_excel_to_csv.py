import pandas as pd

read_file = pd.read_csv(r'C:\Users\tlab\PycharmProjects\transparency-lab\topic_modeling\topic_modeling_LDA_keywords.csv')
read_file.to_excel(r'topic_modeling_LDA_keywords.xlsx', index=None, header=True)

# To csv file
# read_file = pd.read_excel(r'C:\Users\tlab\PycharmProjects\transparency-lab\useful_python_files\papers_divided_by_topic.xlsx')
# read_file.to_csv(r'C:\Users\tlab\PycharmProjects\transparency-lab\useful_python_files\papers_divided_by_topic.csv', index=None, header=True)

