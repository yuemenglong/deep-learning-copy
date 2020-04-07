import F


class ExtractorAuto:
    def __init__(self):
        self.auto = False
        self.right_btn_down = False
        self.last_outer = []
        self.last_landmarks = []
        self.cur_outer = []
        self.cur_landmarks = []
        self.sound_counter = 0

    def loop(self):
        self.sound_counter += 1
        if self.sound_counter % 3000 == 0 and not self.auto:
            F.beep()
