class RailroadRubric:
    def evaluate(self, state, action, next_state):
        # Compositional rewards: safety, procedure, terminology
        return {
            "safety": 0.0,
            "procedure": 0.0,
            "terminology": 0.0
        }
