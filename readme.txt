to change tile:
change n

to change manual input or program input:
find area under commentation manual input, sample program input is provided

type of argument that the checker gives to program:
-   board as string (format and parser included)
-   err0 = wrong input
-   err1 = tile not empty
-   err2 = suicide move (rule violation)

checker expects return in format:
-   string = AlphabetNumber, if the ai wants to move
    example = "a1"
-   string = "pass", if the ai wants to pass

timeout = 5s
if a program timesout, turn is counted as pass

dummy "AI" program = test.py
test.py only tests true input, err0, and err1
switch to manual input (in "execute" function) to test err2

Conclusions:
- Player who passes 3times first loses
- Checker checks if there are no moves left for next player, then game will be ended (unsure if should continue?).
- Checker will calculate area ownership to decide victory (if the game is ended by checker)




