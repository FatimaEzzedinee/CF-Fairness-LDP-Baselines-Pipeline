import pandas as pd
import numpy as np
# from folktables import ACSIncome, ACSDataSource


# import matplotlib.pyplot as plt
 #To fetch a dataset from the PMLB
# from pmlb import fetch_data
# from utils.data.generate_synth import generate_file_name
# from sklearn.model_selection import train_test_split
#from preprocessing import find_feature_ranges
# from folktables import ACSDataSource, ACSEmploymentpip i


dataset_dict = {
    1 : 'adult',
    2 : 'new_adult',
    3 : 'default_credit',
    4 : 'marketing',
    5:  'synth_adult',
    6:  'compas',
    7: 'newadults'
} 

def generate_file_name(datasetname = 'adult', epsilon = 1):
    if epsilon >= 1:
        epsilon = int(epsilon)
    file_name = 'dpnice/datasets/synth_'+str(datasetname) + '_' + str(epsilon) +'.csv'
    return file_name

class Fetcher:
    def __init__(self,name,epsilon = 1000):
        #dataset = fetch_data(name)
        # first: we have only adult dataset
        #adult = fetch_data('adult')
        if name == 'adult':
            dataset = pd.read_csv('dpnice/datasets/adult.csv', sep=',', header=[0], on_bad_lines='skip')
            dataset=dataset.dropna()
            dataset.reset_index(drop=True, inplace=True)

            # dataset = fetch_data(name)   #adult_ds size: 48842 rows
            #Drop unnecessary columns for feature set
            X = dataset.drop(columns=['education-num','fnlwgt','target','native-country','capital-gain','capital-loss'])
            #Assign ‘target’ column with all rows as labels
            y = dataset.loc[:,'target']
            
            #List of columns remaining in X will form the feature set
            feature_names = list(X.columns)
            #assign cost based on how hard the change in feature value is
            costs = [.05,.05,.1,5,.1,5,10,10,.01,.01,.05]
            
            zerocosts = [0] * len(feature_names)
            print(feature_names)
            numerical_features,categorical_features = self.get_feature_types(X)
            #categorical_features = [1,2,3,4,5,6,7]
            #numerical_features = [0,8]
            changhable_features = [0,1,2,4,8]
            increase_only = [0,1,2]
#            costs = [.05,.05,.1,5,.1,5,10,10,.01,.01,.05]

            # X = X.values  
            # print(X.shape)   #Display updated X 

            # y=y.values
            # print(y.shape)   #Display updated y 
            feature_ranges = self.find_feature_ranges(X,numerical_features,categorical_features)
        elif name == 'synth_adult':
            # first generate synthetic data
            # next open correct dataset based on epsilon
            #### OR ######
            #have all data generated and then use them
            # NOTE: if we use synth data we do not need dp-nice, only nice is enough
            #dp_ds('adult',epsilon=epsilon)
            csv_file = generate_file_name(datasetname='adult', epsilon=epsilon)
            dataset = pd.read_csv(csv_file, sep=',', header=[0], on_bad_lines='skip')
            dataset = dataset.sample(n=48842, ignore_index=True, random_state=0)


            X = dataset.drop(columns=['target','capital-gain','capital-loss'])
            #Assign ‘target’ column with all rows as labels
            y = dataset.loc[:,'target']
            
            #List of columns remaining in X will form the feature set
            #assign cost based on how hard the change in feature value is
            
            costs = [.05,.05,.1,5,.1,5,10,10,.01,.01,.05]
            feature_names = list(X.columns)
            zerocosts = [0] * len(feature_names)
            
            print(feature_names)
            numerical_features,categorical_features = self.get_feature_types(X)
            #categorical_features = [1,2,3,4,5,6,7]
            #numerical_features = [0,8]
            changhable_features = [0,1,2,4,8]
            increase_only = [0,1,2]
            costs = [.05,.05,.1,5,.1,5,10,10,.01,.01,.05]

            # X = X.values  
            # print(X.shape)   #Display updated X 

            # y=y.values
            # print(y.shape)   #Display updated y 
            feature_ranges = self.find_feature_ranges(X,numerical_features,categorical_features)
            


        elif name == 'compas':
            dataset = pd.read_csv('dpnice/datasets/compas.csv', sep=',', header=[0], on_bad_lines='skip')
            dataset=dataset.dropna()
            dataset.reset_index(drop=True, inplace=True)
         
            
            # Edit deleted featurs
            X = dataset.drop(columns=['low_risk'])
            
            y = dataset['low_risk']

            feature_names = list(X.columns)
                        
            #assign cost based on how hard the change in feature value is

            #should UPDATE
            costs = zerocosts = [0] * len(feature_names) # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]#[.05,.05,.1,5,.1,5,10,10,.01,.01,.05]
            
            zerocosts = [0] * len(feature_names) #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            print(feature_names)
            
            
            ###### Note: here for now we set this arrays manually
            ###### We should add 
            numerical_features,categorical_features = self.get_feature_types(X)
            
            ##### NOTE: Should Update
            changhable_features = [1,3,4,5,6,7] # we consider Gender, Race, Ethnicity and Birth Weight as unchangable
            increase_only = [1]

            feature_ranges = self.find_feature_ranges(X,numerical_features,categorical_features)
            
        elif name == 'synth_compas':
            csv_file = generate_file_name(datasetname='compas', epsilon=epsilon)
            dataset = pd.read_csv(csv_file, sep=',', header=[0], on_bad_lines='skip')
            # dataset = dataset.sample(n=48842, ignore_index=True, random_state=0)
            
            # dataset = pd.read_csv('dpnice/datasets/compas.csv', sep=',', header=[0], on_bad_lines='skip')
            dataset=dataset.dropna()
            dataset.reset_index(drop=True, inplace=True)
         
            
            # Edit deleted featurs
            X = dataset.drop(columns=['low_risk'])
            
            y = dataset['low_risk']

            feature_names = list(X.columns)
                        
            #assign cost based on how hard the change in feature value is

            #should UPDATE
            costs = zerocosts = [0] * len(feature_names) # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]#[.05,.05,.1,5,.1,5,10,10,.01,.01,.05]
            
            zerocosts = [0] * len(feature_names) #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            print(feature_names)
            
            
            ###### Note: here for now we set this arrays manually
            ###### We should add 
            numerical_features,categorical_features = self.get_feature_types(X)
            
            ##### NOTE: Should Update
            changhable_features = [1,3,4,5,6,7] # we consider Gender, Race, Ethnicity and Birth Weight as unchangable
            increase_only = [1]

            feature_ranges = self.find_feature_ranges(X,numerical_features,categorical_features)
            
        elif name == 'default_credit':
            dataset = pd.read_csv('dpnice/datasets/default_credit.csv', sep=',', header=[0], on_bad_lines='skip')
            dataset=dataset.dropna()
            dataset.reset_index(drop=True, inplace=True)
            
            X = dataset.drop(columns=['DEFAULT_PAYEMENT'])
            y = dataset['DEFAULT_PAYEMENT']

            feature_names = list(X.columns)
            
            
            #should UPDATE
            costs = zerocosts = [0] * len(feature_names) # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]#[.05,.05,.1,5,.1,5,10,10,.01,.01,.05]
            
            zerocosts = [0] * len(feature_names) #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            print(feature_names)
            
            
            ###### Note: here for now we set this arrays manually
            ###### We should add 
            numerical_features,categorical_features = self.get_feature_types(X)
            ##### NOTE: Should Update
            changhable_features = [0,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23] # we consider Gender, Race, Ethnicity and Birth Weight as unchangable
            increase_only = [4]
        

            feature_ranges = self.find_feature_ranges(X,numerical_features,categorical_features)
        
        elif name == 'synth_default_credit':
            csv_file = generate_file_name(datasetname='default_credit', epsilon=epsilon)
            dataset = pd.read_csv(csv_file, sep=',', header=[0], on_bad_lines='skip')
            # dataset = pd.read_csv('dpnice/datasets/default_credit.csv', sep=',', header=[0], on_bad_lines='skip')
            dataset=dataset.dropna()
            dataset.reset_index(drop=True, inplace=True)
            
            X = dataset.drop(columns=['DEFAULT_PAYEMENT'])
            y = dataset['DEFAULT_PAYEMENT']

            feature_names = list(X.columns)
            
            
            #should UPDATE
            costs = zerocosts = [0] * len(feature_names) # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]#[.05,.05,.1,5,.1,5,10,10,.01,.01,.05]
            
            zerocosts = [0] * len(feature_names) #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            print(feature_names)
            
            
            ###### Note: here for now we set this arrays manually
            ###### We should add 
            numerical_features,categorical_features = self.get_feature_types(X)
            ##### NOTE: Should Update
            changhable_features = [0,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23] # we consider Gender, Race, Ethnicity and Birth Weight as unchangable
            increase_only = [4]
        

            feature_ranges = self.find_feature_ranges(X,numerical_features,categorical_features)
            

        elif name == 'hospital':
            dataset = pd.read_csv('dpnice/datasets/hospital_discharge.csv', sep=',', header=[0], on_bad_lines='skip')
            dataset=dataset.dropna()
            dataset.reset_index(drop=True, inplace=True)
            #### next 3 lines should be commented
            # dataset['Length of Stay']=pd.to_numeric(dataset['Length of Stay'], errors ='coerce').fillna(120).astype('int')
            # dataset['Total Costs'] = dataset['Total Costs'].replace('[\$,]', '', regex=True).astype(float)
            # dataset['Total Charges'] = dataset['Total Charges'].replace('[\$,]', '', regex=True).astype(float)

            dataset = dataset.sample(n=48842, ignore_index=True, random_state=0)
            
            # #X = dataset.drop(columns=['Health Service Area','Discharge Year', 'Hospital County','Operating Certificate Number', 'Facility Id', 'Facility Name', 'CCS Diagnosis Code', 'CCS Procedure Code',  'APR DRG Code','APR MDC Code','APR Severity of Illness Code','Total Charges', 'Total Costs'])
            # Edit deleted featurs
            X = dataset.drop(columns=['idx','Health Service Area','Discharge Year', 'Hospital County','Total Charges', 'Total Costs'])
            #X['Zip Code - 3 digits']=X['Zip Code - 3 digits'].astype(str)
            y = dataset['Total Costs'] > np.mean(dataset['Total Costs'])

            feature_names = list(X.columns)
            
            # cat_feat = [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,19,20]
            # num_feat = [5,18]
            # qid = ['Age Group', 'Zip Code - 3 digits', 'Gender', 'Race', 'Ethnicity']
            # target_outcome = True
            # discr_attr = 'Gender'

            # #modelling
            # num=[i for i,v in enumerate(qid) if feature_names.index(v) in num_feat]
            # cat=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]
            # X_train, y_train, X_test,y_test,clf,NICE_fit=model(X,y,feature_names, cat_feat, num_feat, target_outcome)
            
            #assign cost based on how hard the change in feature value is

            #should UPDATE
            costs = zerocosts = [0] * len(feature_names) # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]#[.05,.05,.1,5,.1,5,10,10,.01,.01,.05]
            
            zerocosts = [0] * len(feature_names) #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            print(feature_names)
            
            
            ###### Note: here for now we set this arrays manually
            ###### We should add 
            numerical_features,categorical_features = self.get_feature_types(X)
            #categorical_features = [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,19,20]
            #numerical_features = [5,18]
            ##### NOTE: Should Update
            changhable_features = [0,1,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20] # we consider Gender, Race, Ethnicity and Birth Weight as unchangable
            increase_only = [0]
        

            # X = X.values  
            # print(X.shape)   #Display updated X 

            # y=y.values
            # print(y.shape)   #Display updated y 
            feature_ranges = self.find_feature_ranges(X,numerical_features,categorical_features)
            #print(feature_ranges)

        elif name == 'synth_hospital':
            csv_file = generate_file_name(datasetname='hospital',epsilon=epsilon)
            
            dataset = pd.read_csv(csv_file, sep=',', header=[0], on_bad_lines='skip')
            # dataset=dataset.dropna()
            dataset = dataset.sample(n=48842, ignore_index=True, random_state=0)
            dataset.reset_index(drop=True, inplace=True)

            
            # Edit deleted featurs
            X = dataset.drop(columns=['idx','Health Service Area','Discharge Year', 'Hospital County','Total Charges', 'Total Costs'])
            y = dataset['Total Costs'] > np.mean(dataset['Total Costs'])

            feature_names = list(X.columns)
            
            # cat_feat = [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,19,20]
            # num_feat = [5,18]
            # qid = ['Age Group', 'Zip Code - 3 digits', 'Gender', 'Race', 'Ethnicity']
            # target_outcome = True
            # discr_attr = 'Gender'

            # #modelling
            # num=[i for i,v in enumerate(qid) if feature_names.index(v) in num_feat]
            # cat=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]
            # X_train, y_train, X_test,y_test,clf,NICE_fit=model(X,y,feature_names, cat_feat, num_feat, target_outcome)
            
            #assign cost based on how hard the change in feature value is

            #should UPDATE
            costs = zerocosts = [0] * len(feature_names) # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]#[.05,.05,.1,5,.1,5,10,10,.01,.01,.05]
            
            zerocosts = [0] * len(feature_names) #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            print(feature_names)
            
            
            ###### Note: here for now we set this arrays manually
            ###### We should add 
            numerical_features,categorical_features = self.get_feature_types(X)
            #categorical_features = [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,19,20]
            #numerical_features = [5,18]
            ##### NOTE: Should Update
            changhable_features = [0,1,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20] # we consider Gender, Race, Ethnicity and Birth Weight as unchangable
            increase_only = [0]
        

            # X = X.values  
            # print(X.shape)   #Display updated X 

            # y=y.values
            # print(y.shape)   #Display updated y 
            feature_ranges = self.find_feature_ranges(X,numerical_features,categorical_features)
            #print(feature_ranges)



        elif name == 'informs':
            # url = 'https://github.com/kaylode/k-anonymity/blob/main/data/informs/informs.csv'
            # url = 'https://raw.githubusercontent.com/kaylode/k-anonymity/main/data/informs/informs.csv'
            dataset = pd.read_csv('dpnice/datasets/informs.csv', sep=';', header=[0], on_bad_lines='skip')
            
            dataset.dropna(inplace=True)
            #dataset = dataset.sample(n=5000, ignore_index=True, random_state=0)
            dataset = dataset.sample(n=48842, ignore_index=True, random_state=0)

            dataset.head()
            X = dataset.drop(columns=['income', 'DUID', 'PID', 'ID','RACEAX','RACEBX','RACEWX','RACETHNX'])
            y = dataset['income'] > np.mean(dataset['income'])

            feature_names = list(X.columns)
            #print(feature_names)
            # cat_feat = [2, 3, 4, 5,7,8,9]
            # num_feat = [0, 1, 6]
            
            
            
            #zerocosts = [0,0,0,0,0,0,0,0,0,0]
            print(feature_names)
            numerical_features,categorical_features = self.get_feature_types(X)
            #categorical_features = [1,2,3,4,5,6,7]
            #numerical_features = [0,8]
            changhable_features = [0,1,4,5,6,7,9]  # we consider poverty as changable feature # we consider dobmm and dobyy as changable, since they provide the age and it is changable
            increase_only = []
            ##### to be assigned
            costs = []
            zerocosts = [0] * len(feature_names)#[0,0,0,0,0,0,0,0,0,0]
            

            # X = X.values  
            # print(X.shape)   #Display updated X 

            # y=y.values
            # print(y.shape)   #Display updated y 
            feature_ranges = self.find_feature_ranges(X,numerical_features,categorical_features)

            
            
            #qid = ['DOBMM', 'DOBYY', 'SEX', 'EDUCYEAR', 'marry','RACEX']
            #target_outcome = True
            #discr_attr = 'RACEX'

            #modelling
            #num=[i for i,v in enumerate(qid) if feature_names.index(v) in num_feat]
            #cat=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]
        elif name == 'synth_informs':
            csv_file = generate_file_name(datasetname='informs',epsilon=epsilon)
            
            dataset = pd.read_csv(csv_file, sep=',', header=[0], on_bad_lines='skip')
            dataset = dataset.sample(n=48842, ignore_index=True, random_state=0)
            # dataset=dataset.dropna(inplace=True)
            

            # dataset.head()
            X = dataset.drop(columns=['income', 'DUID', 'PID', 'ID','RACEAX','RACEBX','RACEWX','RACETHNX'])
            y = dataset['income'] > np.mean(dataset['income'])

            feature_names = list(X.columns)
            #print(feature_names)
            # cat_feat = [2, 3, 4, 5,7,8,9]
            # num_feat = [0, 1, 6]
            
            
            
            #zerocosts = [0,0,0,0,0,0,0,0,0,0]
            print(feature_names)
            numerical_features,categorical_features = self.get_feature_types(X)
            #categorical_features = [1,2,3,4,5,6,7]
            #numerical_features = [0,8]
            changhable_features = [0,1,4,5,6,7,9]  # we consider poverty as changable feature # we consider dobmm and dobyy as changable, since they provide the age and it is changable
            increase_only = []
            ##### to be assigned
            costs = []
            zerocosts = [0] * len(feature_names)#[0,0,0,0,0,0,0,0,0,0]
            

            # X = X.values  
            # print(X.shape)   #Display updated X 

            # y=y.values
            # print(y.shape)   #Display updated y 
            feature_ranges = self.find_feature_ranges(X,numerical_features,categorical_features)

            
            
            #qid = ['DOBMM', 'DOBYY', 'SEX', 'EDUCYEAR', 'marry','RACEX']
            #target_outcome = True
            #discr_attr = 'RACEX'

            #modelling
            #num=[i for i,v in enumerate(qid) if feature_names.index(v) in num_feat]
            #cat=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]
        elif name == 'heloc':
            dataset = pd.read_csv('dpnice/datasets/heloc.csv', sep=',', header=[0], on_bad_lines='skip')
            dataset=dataset.dropna()
            dataset.reset_index(drop=True, inplace=True)
            
            X = dataset.drop(columns=['RiskPerformance'])
            y = dataset['RiskPerformance'] #.apply(lambda x: 1 if x == 'Good' else 0)

            feature_names = list(X.columns)
            
            
            #should UPDATE
            costs = zerocosts = [0] * len(feature_names) # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]#[.05,.05,.1,5,.1,5,10,10,.01,.01,.05]
            
            zerocosts = [0] * len(feature_names) #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            print(feature_names)
            
            
            ###### Note: here for now we set this arrays manually
            ###### We should add 
            numerical_features,categorical_features = self.get_feature_types(X)
            ##### NOTE: Should Update
            changhable_features = [0,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23] # we consider Gender, Race, Ethnicity and Birth Weight as unchangable
            increase_only = [4]
        

            feature_ranges = self.find_feature_ranges(X,numerical_features,categorical_features)
        elif name == 'acs_income':  #### it is adult_contrustion table from folktabless
            # year = 2018
            # state = 'CA'
            
            # data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
            # # acs_data = data_source.get_data(states=[state], download=True)
            # acs_data = data_source.get_data(states=[state], download=False)
            # features, label, group = ACSIncome.df_to_numpy(acs_data)
            # # Convert to DataFrame for easier manipulation
            # X = pd.DataFrame(features, columns=ACSIncome.features)
            # y = pd.Series(label, name='income')
            
            # dataset = pd.concat([X, y], axis=1)
            # dataset['income'] = dataset['income'].astype(int)
            # dataset.to_csv('dpnice/datasets/acs_income.csv', index=False)
            # print(X.head())
            # print(y.head())
            dataset = pd.read_csv('dpnice/datasets/acs_income.csv', sep=',', header=[0], on_bad_lines='skip')
            dataset=dataset.dropna()
            dataset.reset_index(drop=True, inplace=True)

            # dataset = fetch_data(name)   #adult_ds size: 48842 rows
            #Drop unnecessary columns for feature set
            X = dataset.drop(columns=['income'])
            #Assign ‘target’ column with all rows as labels
            y = dataset.loc[:,'income']
            
            #List of columns remaining in X will form the feature set
            feature_names = list(X.columns)
            #assign cost based on how hard the change in feature value is
            costs = [.05,.05,.1,5,.1,5,10,10,.01,.01,.05]
            
            zerocosts = [0] * len(feature_names)
            print(feature_names)
            numerical_features,categorical_features = self.get_feature_types(X)
            #categorical_features = [1,2,3,4,5,6,7]
            #numerical_features = [0,8]
            changhable_features = [0,1,2,4,8]
            increase_only = [0,1,2]
#            costs = [.05,.05,.1,5,.1,5,10,10,.01,.01,.05]

            # X = X.values  
            # print(X.shape)   #Display updated X 

            # y=y.values
            # print(y.shape)   #Display updated y 
            feature_ranges = self.find_feature_ranges(X,numerical_features,categorical_features)
        elif name == 'synth_folktables':
            # first generate synthetic data
            # next open correct dataset based on epsilon
            #### OR ######
            #have all data generated and then use them
            # NOTE: if we use synth data we do not need dp-nice, only nice is enough
            #dp_ds('adult',epsilon=epsilon)
            csv_file = generate_file_name(datasetname='folktables', epsilon=epsilon)
            dataset = pd.read_csv(csv_file, sep=',', header=[0], on_bad_lines='skip')
            dataset = dataset.sample(n=48842, ignore_index=True, random_state=0)


            X = dataset.drop(columns=['target','capital-gain','capital-loss'])
            #Assign ‘target’ column with all rows as labels
            y = dataset.loc[:,'target']
            
            #List of columns remaining in X will form the feature set
            #assign cost based on how hard the change in feature value is
            
            costs = [.05,.05,.1,5,.1,5,10,10,.01,.01,.05]
            feature_names = list(X.columns)
            zerocosts = [0] * len(feature_names)
            
            print(feature_names)
            numerical_features,categorical_features = self.get_feature_types(X)
            #categorical_features = [1,2,3,4,5,6,7]
            #numerical_features = [0,8]
            changhable_features = [0,1,2,4,8]
            increase_only = [0,1,2]
            costs = [.05,.05,.1,5,.1,5,10,10,.01,.01,.05]

            # X = X.values  
            # print(X.shape)   #Display updated X 

            # y=y.values
            # print(y.shape)   #Display updated y 
            feature_ranges = self.find_feature_ranges(X,numerical_features,categorical_features)

        else:
            print("No valid dataset assigned")
        

        
        self.dataset = {
            'X' : X.astype(np.float32),
            'y' : y,
            'feature_names':feature_names,
            'categorical_features' : categorical_features,
            'numerical_features' : numerical_features,
            'changhable_features' :changhable_features,
            'increase_only' : increase_only,
            'feature_ranges' : feature_ranges,
            'costs' : costs,
            'zerocosts' : zerocosts,
            'change_cost' : costs,
            'X_test' : [],
            'y_test':[]
            }
        

    def generate_file_name(datasetname = 'adult', epsilon = 1):
        if epsilon >= 1:
            epsilon = int(epsilon)
        file_name = 'dpnice/datasets/synth_'+str(datasetname) + '_' + str(epsilon) +'.csv'
        return file_name
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=RANDOM_SEED)
#needa more investigation to find categorical features
    def get_feature_types(self,X):
            feature_names = list(X.columns)
            X_na= X.dropna()
            con_feat = []
            cat_feat =[]
            idx = 0
            for feature_name in feature_names:
                x = X_na[feature_name].copy()
                if  not all(float(i).is_integer() for i in x.unique()):
                    #con_feat.append(feature_name)
                    con_feat.append(idx)
                elif x.nunique() > 10:
                    #con_feat.append(feature_name)
                    con_feat.append(idx)
                else:
                    #cat_feat.append(feature_name)
                    cat_feat.append(idx)
                idx +=1
            return con_feat, cat_feat

    def find_feature_ranges(self,dataset,numerical_feature_indices,categorical_feature_indices):
        num_features = len(numerical_feature_indices)
        min_values = [float('inf')] * num_features
        max_values = [float('-inf')] * num_features

        categorical_values = {idx: set() for idx in categorical_feature_indices}

        for record in dataset.values:
            i = 0
            for num_idx in numerical_feature_indices:
                num_value = record[num_idx]
                min_values[i] = min(min_values[i], num_value)
                max_values[i] = max(max_values[i], num_value)
                i +=1
            i =0
            for cat_idx in categorical_feature_indices:
                cat_value = record[cat_idx]
                categorical_values[cat_idx].add(cat_value)
                i+=1

        result = {
            'numerical': {
                'min': min_values,
                'max': max_values
            },
            'categorical': categorical_values
        }

        return result
