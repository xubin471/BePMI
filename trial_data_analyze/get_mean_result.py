import re
import numpy as np
class Result:
    def __init__(self):
        self.organs = ['LIVER',"RK","LK","SPLEEN"]
        # self.organs = ['LV-MYO', "LV-BP", "RV"]
        # self.organs=['Bladder','Bone','TZ','SV']
        self.content = self.readFile("trial_result")
    def readFile(self,file_path):
        with open(file_path, 'r') as file:
            content = file.read()
        return content

    def extract_values(self,organ_name):
        pattern = fr"'{organ_name}': 0\.(\d{{6}})"
        matches = re.findall(pattern, self.content)
        return matches

    def extract_mean(self):
        mean_organ  = {}
        for organ in self.organs:
            result_folds = np.array([ int(value) for value in self.extract_values(organ)])
            mean = np.mean(result_folds)*1e-4
            mean_organ[organ] = float(f"{mean:.2f}")

        print(mean_organ)

        total_mean = sum(mean_organ.values()) / len(mean_organ)
        total_mean = float(f"{total_mean:.2f}")
        print(f"mean: {total_mean}")


file_path = 'trial_result'
result = Result()
result.extract_mean()
