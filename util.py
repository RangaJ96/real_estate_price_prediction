
class Utilities:

    def is_float(x):
        try:
            float(x)
        except:
            return False
        return True

    def covert_sqft_to_num(x):
        token = x.split('-')
        if len(token) == 2:
            return (float(token[0]) + float(token[1]))/2
        try:
            return float(x)
        except:
            return None
