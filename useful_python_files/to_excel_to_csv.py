import pandas as pd

# read_file = pd.read_csv(r'keywords_white_papers.csv')
# read_file.to_excel(r'keywords_white_papers.xlsx', index=None, header=True)

# To csv file
read_file = pd.read_excel(r'C:\Users\tlab\PycharmProjects\transparency-lab\useful_python_files\papers_divided_by_topic.xlsx')
read_file.to_csv(r'C:\Users\tlab\PycharmProjects\transparency-lab\useful_python_files\papers_divided_by_topic.csv', index=None, header=True)

