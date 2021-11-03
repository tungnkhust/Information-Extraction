from src.schema import Token


def create_token(text):
    token = Token.from_text(text)
    print(token.to_dict())


if __name__ == '__main__':
    text1 = "1	April	B-Other	['N']	[1]"
    text2 = "25	Brigade	I-Org	['OrgBased_In', 'OrgBased_In', 'OrgBased_In']	[14, 40, 8]"

    create_token(text2)