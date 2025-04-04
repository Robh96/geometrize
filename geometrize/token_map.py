class TokenMap:
    def __init__(self):
        self.token_to_id = {
            'SPHERE': 0,
            'CUBOID': 1,
            'CONE': 2,
            'UNION': 3,
            'SUBTRACT': 4,
            'SCALE': 5,
            'TRANSLATE': 6,
            'ROTATE': 7,
            'CENTER_X': 8,
            'CENTER_Y': 9,
            'CENTER_Z': 10,
            'RADIUS': 11,
            'DX': 12,
            'DY': 13,
            'DZ': 14,
            'NORM_X': 15,
            'NORM_Y': 16,
            'NORM_Z': 17,
            'RADIUS_0': 18,
            'RADIUS_1': 19,
            'OBJECT_1': 20,
            'OBJECT_2': 21,
            'OBJECT': 22,
            'SCALE_X': 23,
            'SCALE_Y': 24,
            'SCALE_Z': 25,
            'THETA_X': 26,
            'THETA_Y': 27,
            'THETA_Z': 28,
            'NULL': 29
        }
        self.id_to_token = {id: token for token, id in self.token_to_id.items()}

    def get_id(self, token):
        return self.token_to_id.get(token)

    def get_token(self, id):
        return self.id_to_token.get(id)