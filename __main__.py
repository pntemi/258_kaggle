from framework import run

args = {
    'submission': True,
    'train_fraction': 0.8,
    'random_state': 12345,
    'expt': 'expt_24',
    'cross_val_score': False,
    'k-fold': 3,
    'model': 'Rating'
    # 'model': 'Visit'
}

if __name__ == "__main__":
    run(args)

