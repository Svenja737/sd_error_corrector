from modeling.lex_corrector_model import LexSDCorrector

def main():

    lex_sd = LexSDCorrector()
    train, val, test = lex_sd.prepare_data()
    lex_sd.train_model(train, val)
    # lex_sd.test_model(test)

if __name__ == "__main__":
    main()
    