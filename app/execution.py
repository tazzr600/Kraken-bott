from decimal import Decimal

class PaperExecutor:
    def __init__(self, settings):
        self.settings = settings

    def buy(self, token_id, shares, price, tick_size="0.01"):
        return {"status":"PAPER_FILLED","token_id":str(token_id),"filled_shares":float(shares),"price":float(price)}

    @staticmethod
    def response_id(response): return "PAPER"

    @staticmethod
    def filled_shares(response, fallback):
        if isinstance(response,dict) and response.get("filled_shares") is not None:
            return Decimal(str(response["filled_shares"]))
        return Decimal("0")

    @staticmethod
    def raw(response): return str(response)
