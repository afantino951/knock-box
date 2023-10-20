class DoorStateMachine:
    def __init__(self, allowable_quiet):
        self.state = "start"
        self.correct_sequence = ["knock", "elbow", "knock", "knock"]
        self.sequence_index = 0
        self.quiet_count = 0
        self.quiet_limit = allowable_quiet

    def process_interaction(self, interaction):
        if self.state == "start":
            if interaction == self.correct_sequence[self.sequence_index]:
                self.sequence_index += 1
            elif interaction == "quiet":
                self.quiet_count += 1
                if self.quiet_count > self.quiet_limit:
                    self.sequence_index = 0
                    self.quiet_count = 0
            else:
                self.sequence_index = 0

            if self.sequence_index == len(self.correct_sequence):
                self.state = "success"
        elif self.state == "success":
            print("Door opened!")
            self.state = "start"
            self.sequence_index = 0
            self.quiet_count = 0

# Sample data stream, replace with your actual data stream
# data_stream = ["knock", "quiet", "knock", "quiet", "knock", "quiet", "elbow", "quiet", "knock", "knock", "quiet", "knock"]

# door_state_machine = DoorStateMachine(5)
# for interaction in data_stream:
#     print(interaction)
#     door_state_machine.process_interaction(interaction)
#     print(door_state_machine.state)

# Sample data stream 1: A valid sequence
data_stream1 = ["knock", "quiet", "knock", "quiet", "knock", "quiet", "elbow", "quiet", "knock", "knock", "quiet", "knock"]

# Sample data stream 2: An invalid sequence with too many quiet interactions
data_stream2 = ["knock", "quiet", "knock", "quiet", "knock", "quiet", "elbow", "quiet", "knock", "quiet", "quiet", "quiet", "quiet", "knock"]

# Sample data stream 3: An invalid sequence with a wrong interaction
data_stream3 = ["knock", "quiet", "knock", "knock", "elbow", "knock", "knock"]

# Sample data stream 4: A valid sequence that opens the door
data_stream4 = ["knock", "quiet", "knock", "quiet", "knock", "quiet", "elbow", "quiet", "knock", "knock", "quiet", "knock", "elbow", "knock", "knock"]

# Sample data stream 5: An empty data stream
data_stream5 = []

door_state_machine = DoorStateMachine(5)

data_streams = [data_stream1, data_stream2, data_stream3, data_stream4, data_stream5]

n = 0
for data_stream in data_streams:
    door_state_machine.state = "start"
    door_state_machine.sequence_index = 0
    door_state_machine.quiet_count = 0
    n += 1

    print("Processing data stream:", n)
    for interaction in data_stream:
        door_state_machine.process_interaction(interaction)
    print()

