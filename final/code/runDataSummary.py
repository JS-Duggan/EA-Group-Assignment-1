import pandas
import statistics

def calculateStatistics(data):
    """
        Calcualtes 5 key statistical values for a set of data and displays them on the terminal

        Args:
            data (list[int]): Data through which the calculations are based on

        Returns:
            """
    
    print(' - mean = ' + str(statistics.mean(data)))
    print(' - median = ' + str(statistics.median(data)))
    print(' - mode = ' + str(statistics.mode(data)))

    print(' - max = ' + str(max(data)))
    print(' - min = ' + str(min(data)))

def processData(path):
    """
        Processes the data at the file path and caculates key statistical values for the three different methods
        Results are published on the terminal

        Args:
            path (string): File path to the raw data to be processed

        Returns:
            """
    
    file = pandas.read_csv(path)
    JumpCosts = []
    ExchangeCosts = []
    InverseCosts = []

    # Process the relevant data from each row
    for index, row in file.iterrows():
        JumpCosts.append(int(row['Jump - cost']))
        ExchangeCosts.append(int(row['Exchange - cost']))
        InverseCosts.append(int(row['Inverse - cost']))

    # Calcualte the data and display it to terminal
    print(path + ':')
    print('Jump: ')
    calculateStatistics(JumpCosts)
    print('Exchange: ')
    calculateStatistics(ExchangeCosts)
    print('Inverse: ')
    calculateStatistics(InverseCosts)


# processData('final/code/temp_results/eil51.csv')