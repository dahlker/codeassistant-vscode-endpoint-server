from request_handler import RequestHandler


class RequestHandlerProvider:
    def __init__(self, request_handler: RequestHandler):
        self.request_handler = request_handler

    def get_handler(self) -> RequestHandler:
        return self.request_handler
