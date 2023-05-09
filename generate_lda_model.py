from sp_estimation import create_lda_model
from sp_estimation import load_all_data
import sys

def main():
    if len(sys.argv) <= 1:
        print("Usage: python generate_lda_model.py [PATH TO DATASET] [NUMBER OF TOPICS]")
        print("required: [PATH TO DATASET]")
        print("optional (integer): [NUMBER OF TOPICS]")
        sys.exit(1)

    print("Loading data...")
    data_path = sys.argv[1]
    train = load_all_data(data_path, "train")
    valid = load_all_data(data_path, "valid")

    if len(sys.argv) == 2: # need to find best t; time/compute intensive
        data_path = sys.argv[1]
        # Note:
        # - finding best t involves building many LDA models in the range 15 to 2015 in 250 increments (8 LDA Models)
        # - intermediate models are not saved
        # - chooses t-value based on the model that produces the lowest perplexity score
        # - final LDA model will be built (and then saved) using that t-value
        create_lda_model(train['issue_context'], valid['issue_context'])

    elif len(sys.argv) == 3: # t is given
        data_path = sys.argv[1]
        num_of_topics = sys.argv[2]
        create_lda_model(train['issue_context'], valid['issue_context'], int(num_of_topics))
    

if __name__ == '__main__':
    main()