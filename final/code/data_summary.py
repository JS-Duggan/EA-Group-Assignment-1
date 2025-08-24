import pandas
import statistics

def calculateStatistics(data):
    """
    Calcualtes 5 key statistical values for a set of data and displays them on the terminal

    Inputs:
        data (list[int]): Data through which the calculations are based on
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
    jump_costs = []
    exchange_costs = []
    inverse_costs = []

    # Process the relevant data from each row
    for index, row in file.iterrows():
        jump_costs.append(int(row['Jump - cost']))
        exchange_costs.append(int(row['Exchange - cost']))
        inverse_costs.append(int(row['Inverse - cost']))

    # Calcualte the data and display it to terminal
    print(path + ':')
    print('Jump: ')
    calculateStatistics(jump_costs)
    print('Exchange: ')
    calculateStatistics(exchange_costs)
    print('Inverse: ')
    calculateStatistics(inverse_costs)
