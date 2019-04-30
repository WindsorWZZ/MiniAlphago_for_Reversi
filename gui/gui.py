import wx
from wx.core import CommandEvent
from .env_gui import EnvGui
from env.env import Stone
import time
def start(config):
    app = wx.App()
    Frame(config).Show()
    app.MainLoop()

def notify(caption, message):
    dialog = wx.MessageDialog(None, message=message, caption=caption, style=wx.OK)
    dialog.ShowModal()
    dialog.Destroy()

class Frame(wx.Frame):
    def __init__(self, config):
        # params
        wx.Frame.__init__(self, None, -1, "Othello", size=(500, 500))

        # models
        self.model = EnvGui(config=config)

        # panel
        self.panel = wx.Panel(self)
        self.panel.Bind(wx.EVT_LEFT_DOWN, self._int_try_move)
        self.panel.Bind(wx.EVT_PAINT, self._int_refresh)

        # menu bar
        menu = wx.Menu()
        menu.Append(1, u"New Game(Black)")
        menu.Append(2, u"New Game(White)")
        menu_bar = wx.MenuBar()
        menu_bar.Append(menu, u"MENU")
        self.SetMenuBar(menu_bar)
        self.Bind(wx.EVT_MENU, self._handle_new_game, id=1)
        self.Bind(wx.EVT_MENU, self._handle_new_game, id=2)

        # status bar
        self.CreateStatusBar()

        # args
        self.brushes = {1: wx.Brush("white"), 2: wx.Brush("black")}
    @property
    def __ob(self):
        return {1:self._rev_refresh, 2:self._rev_ai_move, 3:self._rev_game_over}

    def _handle_new_game(self, event: CommandEvent):
        self.model.start_game(human_is_black=event.GetId() == 1)
        self.model.add_observer(self.__ob)
        self.model.play_next_turn()

    def _rev_ai_move(self):
        self.model.move_by_ai()
        self.model.play_next_turn()

    def _int_try_move(self, event):
        w, h = self.panel.GetSize()
        x = int(event.GetX() / (w / 8))
        y = int(event.GetY() / (h / 8))
        if self.model.env.done or not self.model.available(x, y):
            self._rev_wrong_step()
            return
        # do
        self.model.move(x, y)
        self.model.play_next_turn()

    # update everything
    def _rev_refresh(self):
        self.panel.Refresh()
        self.__update_status_bar()
        wx.Yield() # show update

    def _rev_wrong_step(self):
        # if game is over then display dialog
        notify("ERROR", "WRONG STEP")

    def _rev_game_over(self):
        # if game is over then display dialog
        black, white = self.model.env.chessboard.black_white
        head = "RESULT"
        mes = "black: %d\nwhite: %d\n" % (black, white)
        res = "-- draw --" if black == white else "--winner: %s" % ["black", "white"][black < white]+"--"
        notify(head, mes+res)

    def _int_refresh(self, event):
        # parmsa
        self.__update_status_bar()

        # background
        w, h = self.panel.GetSize()
        px, py = w / 8, h / 8
        dc = self.__print_frame(w, h, px, py)

        # stone
        for y in range(8):
            for x in range(8):
                self.__print_stone(x, y, dc, px, py)
                self.__print_history(x, y, dc, px, py)

    def __print_frame(self, w, h ,px ,py):
        dc = wx.PaintDC(self.panel)
        dc.SetBrush(wx.Brush("#808080"))
        dc.DrawRectangle(0, 0, w, h)
        dc.SetBrush(wx.Brush("black"))
        for y in range(8):
            dc.DrawLine(y * px, 0, y * px, h)
            dc.DrawLine(0, y * py, w, y * py)
        dc.DrawLine(w - 1, 0, w - 1, h - 1)
        dc.DrawLine(0, h - 1, w - 1, h - 1)
        return dc

    def __print_stone(self, x, y, dc, px, py):
        c = self.model.stone(x, y)
        if c:
            dc.SetBrush(self.brushes[c])
            dc.DrawEllipse(x * px, y * py, px, py)
            dc.SetPen(wx.Pen("808080"))
            dc.DrawEllipse(x * px - 1, y * py - 1, px + 2, py + 2)
        else:
            if self.model.last_ava:
                ava = self.model.last_ava.current
                enemy_ava = self.model.last_ava.next
                a = ava & (1 << (y * 8 + x))
                b = enemy_ava & (1 << (y * 8 + x))
                if b:
                    dc.SetBrush(wx.Brush("#808080"))
                    dc.SetPen(wx.RED_PEN)
                    dc.DrawEllipse(x * px - 1, y * py - 1, px + 2, py + 2)
                if a:
                    dc.SetBrush(wx.Brush("#808080"))
                    dc.SetPen(wx.BLUE_PEN)
                    dc.DrawEllipse(x * px + 5, y * py + 5 , px - 10, py - 10)


    def __print_history(self, x, y ,dc ,px ,py):
        if self.model.last_history:
            n_value = self.model.last_history.visit[y * 8 + x]
            q_value = self.model.last_history.values[y * 8 + x]
            enemy_n_value = self.model.last_history.enemy_visit[y * 8 + x]
            enemy_q_value = - self.model.last_history.enemy_values[y * 8 + x]
            # foreground
            dc.SetTextForeground(wx.Colour("blue"))
            if n_value:
                dc.DrawText(f"{int(n_value):d}", x * px + 2, y * py + 2)
            if q_value:
                if q_value < 0:
                    dc.SetTextForeground(wx.Colour("red"))
                dc.DrawText(f"{q_value:3f}", x * px + 2, (y + 1) * py - 16)
            # foreground
            dc.SetTextForeground(wx.Colour("purple"))
            if enemy_n_value:
                dc.DrawText(f"{int(enemy_n_value):2d}", (x + 1) * px - 20, y * py + 2)
            if enemy_q_value:
                if enemy_q_value < 0:
                    dc.SetTextForeground(wx.Colour("orange"))
                dc.DrawText(f"{enemy_q_value:3f}", (x + 1) * px - 24, (y + 1) * py - 16)


    def __update_status_bar(self):
        msg = "wait to play:"+["white", "black"][self.model.env.next_to_play == Stone.black] + '\n'
        if self.model.last_evaluation:
            msg += f"|value={self.model.last_evaluation:.2f}\n"
        msg += f'|this={self.model.count_one_step:.1f}\n'
        msg += f'|all={self.model.count_all_step:.1f}\n'
        msg += f"|action={self.model.action}"
        self.SetStatusText(msg)

