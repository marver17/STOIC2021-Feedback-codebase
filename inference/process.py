from typing import Dict
from pathlib import Path
import SimpleITK
import pickle


from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

from algorithm.preprocess import extract_all_
from utils import MultiClassAlgorithm


COVID_OUTPUT_NAME = Path("probability-covid-19")
SEVERE_OUTPUT_NAME = Path("probability-severe-covid-19")


class StoicAlgorithm(MultiClassAlgorithm):

    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path=Path("/input/images/ct/"),
            output_path=Path("/output/")
        )
        # load model

        filename = "./artifact/model_c.sav"
        filename_1 = "./artifact/model_s.sav"
        self.model_C = pickle.load(open(filename, 'rb'))
        self.model_S = pickle.load(open(filename_1, 'rb'))






    def predict(self, *,input_images : SimpleITK.Image,input_path :str) -> Dict:
        # pre-processing
        feautures,check  = extract_all_(input_images, input_path)   #chiamarlo feautures
        # filename = "./algorithm/modello_covid.pth"
        # filename_1 = "./algorithm/modello_severo.pth"
        if check == 0:
            prob_covid =  [[0.0, 0.0]]
            prob_severe = [[0.0, 0.0]]
        else :
            # # run model
            # prob_covid = infer(feautures,filename)
            # prob_severe = infer(feautures,filename_1)
            # print(prob_covid[0])
            # print(prob_severe[0])
            feautures = feautures.reshape(1, -1)
            print("OKKK")
            # feautures_covid = feautures_covid.reshape(1, -1)

            # # run model
            prob_covid = self.model_C.predict_proba(feautures)
            prob_severe = self.model_S.predict_proba(feautures)
            print(prob_covid[0][1])
            print(prob_severe[0][1])
        return {
            COVID_OUTPUT_NAME:  float(prob_covid[0][1]),
            SEVERE_OUTPUT_NAME: float(prob_severe[0][1])
        }


if __name__ == "__main__":
    StoicAlgorithm().process()
