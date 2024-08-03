def get_model(model_name, args):
    name = model_name.lower()
    if name == "aisp":
        from models.aisp import Learner
    else:
        assert 0
    return Learner(args)
