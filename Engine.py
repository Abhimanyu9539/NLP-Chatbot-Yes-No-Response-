## Import required libraries
from preprocessing import preprocess, extract_tagged, extract_feature_from_doc,extract_feature,word_feats
from models import nbclassifier, dtclassifier,answers
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

## Main function to get the predictions
def main(input_sentence):
    category = nbclassifier.classify(word_feats(extract_feature(input_sentence)))
    return answers[category]
    
if __name__ == "__main__":   
    #print("In main function")

    while True:
        txt = input().lower()
        if txt == "bye" or txt == "goodbye":
        
            print("| Bot: Bye")
            exit()

        else:
            if txt == "i want to apply for loan." or txt == "i want to apply for loan":
                print("| Bot: ")
            else:
                print("| Bot:",main(txt))

    