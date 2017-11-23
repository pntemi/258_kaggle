from framework import run

args = {
    'submission': True,
    'test_size': 0.2,
    'random_state': 12345,
    'expt': 'expt_4',
    'cross_val_score': False,
    'k-fold': 3,
    'model': 'Rating'
    # 'model': 'Visit'
}

if __name__ == "__main__":
    run(args)

