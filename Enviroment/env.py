from .Side.server import Server

class Enviroment:
    def __init__(self, env_ip="0.0.0.0", env_port=5060):
        self.server = Server(env_ip, env_port)

        self.current_state = None

    def step(self, action : list[list[int]]) -> tuple[list[int], list[int]]:
        """
        tuple[CharacterBonesTransformValues, RGBValues]
        """
        self.server.send_to_client(action)

        self.current_state = self.server.receive_from_client()
        return self.current_state

    #sadece resetlendiğinde Goal döndürülüyor
    def reset(self) -> tuple[tuple[list[int], list[int]], list]:
        """
        tuple[tuple[CharacterBonesTransformValues, RGBValues], Goal]
        """
        self.server.send_to_client([[0,0,0]])

        *self.current_state, self.goal = self.server.receive_from_client()
        return self.current_state, self.goal

    def close(self):
        self.server.send_to_client([[-1,-1,-1]], control_receive=False)
        self.server.disconnect(keep_server=False)
        

if __name__ == "__main__":
    env = Enviroment()

    import random
    env.reset() 
    for i in range(150):
        action = []
        for i in range(31):
            bt = random.sample(range(1, 100), 3)
            action.append(bt)
        env.step(action)

    env.close()




    