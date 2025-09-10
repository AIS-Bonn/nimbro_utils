#!/usr/bin/env python3

import numpy as np

from nimbro_utils.utility.misc import assert_type_value

class Color:
    """
    Represents a single named color with multiple color formats.

    Attributes:
        name (str): The name of the color (e.g. "red").
        hex (str): The hexadecimal color code (e.g. "#FF0000").
        rgb (Tuple[int, int, int]): The 8bit RGB representation as a tuple of integers.
        bgr (Tuple[int, int, int]): The 8bit BGR representation as a tuple of integers.
    """

    def __init__(self, hex_code, name="Color"):
        """
        Initializes a Color instance.

        Args:
            hex_code (str): The hex code of the color (e.g. "#AABBCC").
            name (str): The name of the color.

        Raises:
            AssertionError: If input arguments are invalid.
        """
        # parse arguments
        assert_type_value(obj=hex_code, type_or_value=str, name="argument 'hex_code'")
        assert_type_value(obj=name, type_or_value=str, name="argument 'name'")
        assert name not in ["name", "hex", "rgb", "bgr"], f"Color name refers to reserved keyword '{name}'."

        self.name = name
        self.hex = hex_code
        self.rgb = self._hex_to_rgb(hex_code)
        self.bgr = self.rgb[::-1]

    def _hex_to_rgb(self, hex_code):
        hex_code = hex_code.lstrip("#")
        return tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))

    def __repr__(self):
        return f"{self.name}(hex='{self.hex}', rgb={self.rgb}, bgr={self.bgr})"

class ColorPalette:
    """
    A collection of named colors with optional named subgroups.

    Colors and subgroups can be accessed as attributes or by key.

    Attributes:
        name (str): Name of the palette.
        names (Tuple[str]): All color names in the palette.
        hex (Tuple[str]): All hex codes in the palette.
        hex_shuffle (Tuple[str]): All hex codes in the palette in random order.
        rgb (Tuple[Tuple[int, int, int]]): All 8bit RGB tuples in the palette.
        rgb_shuffle (Tuple[Tuple[int, int, int]]): All 8bit RGB tuples in the palette in random order.
        bgr (Tuple[Tuple[int, int, int]]): All 8bit BGR tuples in the palette.
        bgr_shuffle (Tuple[Tuple[int, int, int]]): All 8bit BGR tuples in the palette in random order.

    Raises:
            AssertionError: If input arguments are invalid.

    Example:
        palette = ColorPalette({
            "red": "#FF0000",
            "green": "#00FF00",
            "blue": "#0000FF"
        }, groups={"primary": ["red", "blue"]})

        palette.red.rgb      # (255, 0, 0)
        palette.primary.hex  # ("#FF0000", "#0000FF")
    """

    def __init__(self, colors, name="ColorPalette", groups=None):
        """
        Initialize a ColorPalette.

        Args:
            colors (dict[str, str]): Mapping of color names to hex codes.
            name (str, optional): Name of the ColorPalette.
            groups (dict[str, list[str]] | None, optional): Mapping of group names to lists of color names.

        Raises:
            AssertionError: If input arguments are invalid.
        """
        # parse arguments
        assert_type_value(obj=colors, type_or_value=dict, name="argument 'colors'")
        assert_type_value(obj=name, type_or_value=str, name="argument 'name'")
        assert_type_value(obj=groups, type_or_value=[dict, None], name="argument 'groups'")

        self.name = name
        self.names = []
        self._colors = []
        assert len(colors) > 0, "Palette must at least contain one color."
        for name, hex_code in colors.items():
            assert name not in ["name", "names", "hex", "rgb", "bgr", "groups"], f"Color name refers to reserved keyword '{name}'."
            self.names.append(name)
            self._colors.append(Color(hex_code=hex_code, name=name))
            setattr(self, name, self._colors[-1])
        self.names = tuple(self.names)
        self._colors = tuple(self._colors)

        self.groups = {}
        if groups:
            for group_name, group_keys in groups.items():
                assert group_name not in ["name", "names", "hex", "rgb", "bgr"], f"Group name refers to reserved keyword '{group_name}'."
                assert group_name not in colors, f"Group name refers to known color '{group_name}'."
                for key in group_keys:
                    assert key in colors, f"Group name '{group_name}' refers to unknown color '{key}'."
                group_dict = {k: colors[k] for k in group_keys}
                subgroup = ColorPalette(colors=group_dict, name=f"{self.name}.{group_name}")
                setattr(self, group_name, subgroup)
                self.groups[group_name] = tuple(group_keys)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._colors[key]
        elif isinstance(key, str):
            if hasattr(self, key):
                return getattr(self, key)
        raise KeyError(f"Invalid key: {key}")

    @property
    def hex(self):  # noqa: A003
        return tuple(c.hex for c in self._colors)

    @property
    def hex_shuffle(self):
        colors = [c.hex for c in self._colors]
        np.random.shuffle(colors)
        return tuple(colors)

    @property
    def rgb(self):
        return tuple(c.rgb for c in self._colors)

    @property
    def rgb_shuffle(self):
        colors = [c.rgb for c in self._colors]
        np.random.shuffle(colors)
        return tuple(colors)

    @property
    def bgr(self):
        return tuple(c.bgr for c in self._colors)

    @property
    def bgr_shuffle(self):
        colors = [c.bgr for c in self._colors]
        np.random.shuffle(colors)
        return tuple(colors)

    def __len__(self):
        return len(self.names)

    def __repr__(self):
        if len(self.groups) > 0:
            colors = "\n\t\t" + ",\n\t\t".join(repr(c) for c in self._colors) + "\n\t"
            groups = "\n\t\t" + ",\n\t\t".join(f"{name}{repr(self.groups[name]).replace("'", "")}" for name in self.groups) + "\n\t" # noqa: E999
            return f"{self.name}(\n\tcolors: [{colors}],\n\tgroups: [{groups}]\n)"
        else:
            colors = "\n\t" + ",\n\t".join(repr(c) for c in self._colors) + "\n"
            return f"{self.name}([{colors}])"

def show_colors(colors, pixels=100, columns=25, shuffle=False):
    """
    Display a grid of color patches from a ColorPalette or list of BGR/hex colors.

    Raises:
        AssertionError: If input arguments are invalid.

    Args:
        colors (ColorPalette | list[tuple[int, int, int]] | list[str]):
            The input colors to display. Can be a ColorPalette,
            a list of BGR tuples, or a list of hex strings.
        pixels (int, optional): Size of each patch (pixels x pixels). Defaults to 100.
        columns (int, optional): Number of patches per row. Defaults to 25.
        shuffle (bool, optional): Whether to shuffle the colors before displaying. Defaults to False.
    """
    # parse arguments
    assert_type_value(obj=colors, type_or_value=[ColorPalette, list, tuple], name="argument 'colors'")
    assert_type_value(obj=pixels, type_or_value=int, name="argument 'pixels'")
    assert_type_value(obj=columns, type_or_value=int, name="argument 'columns'")
    assert_type_value(obj=shuffle, type_or_value=bool, name="argument 'shuffle'")

    if isinstance(colors, ColorPalette):
        colors = colors.bgr

    if shuffle is True:
        colors = list(colors)
        np.random.shuffle(colors)

    columns = min(columns, len(colors))

    def hex_to_bgr(h):
        h = h.lstrip('#')
        rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        bgr = (rgb[2], rgb[1], rgb[0])
        return bgr

    image = None
    for i, color in enumerate(colors):
        patch = np.full(
            shape=(pixels, pixels, 3),
            fill_value=hex_to_bgr(colors[i]) if isinstance(colors[i], str) else colors[i],
            dtype=np.uint8
        )
        if i % columns == 0:
            row = patch
        else:
            row = np.concatenate((row, patch), axis=1)
        if i % columns == columns - 1:
            if image is None:
                image = row
            else:
                image = np.concatenate((image, row), axis=0)

    fill = columns - (i % columns) - 1
    if fill > 0:
        for i in range(fill):
            patch = np.full(shape=(pixels, pixels, 3), fill_value=(0, 0, 0), dtype=np.uint8)
            row = np.concatenate((row, patch), axis=1)
        if image is None:
            image = row
        else:
            image = np.concatenate((image, row), axis=0)

    from nimbro_utils.utility.image import show_image
    show_image(image)

nimbro = ColorPalette(
    colors={
        'petrol': "#0C5678",
        'sun': "#FFB500",
        'violet': "#412163",
        'lime': "#9AB800",
        'red': "#900000",
        'sky': "#92B7CF",
        'teal': "#145050",
        'rose': "#FFD7B6",
        'salmon': "#E37092",
        'khaki': "#8C8240",
        'purple': "#B799FF",
        'surf': "#A3D590",
        'yellow': "#FFEA00",
        'blue': "#3975AC",
        'rosa': "#FFCADE",
        'cyan': "#009996",
        'blood': "#4F000A",
        'olive': "#455200",
        'pink': "#DA0050",
        'indigo': "#000D80",
        'orange': "#FF7300",
        'green': "#457400",
        'lila': "#8200BE",
        'mint': "#D2F5CC",
        'brown': "#663C00"
    },
    name="nimbro",
    groups={'ten': ['petrol', 'sun', 'violet', 'lime', 'red', 'sky', 'teal', 'rose', 'salmon', 'khaki']}
)

kelly = ColorPalette(
    colors={
        'white': "#FdFDFD",
        'black': "#1D1D1D",
        'yellow': "#EBCE2B",
        'purple': "#702C8C",
        'orange': "#DB6917",
        'aqua': "#96CDE6",
        'red': "#BA1C30",
        'buff': "#C0BD7F",
        'gray': "#7F7E80",
        'green': "#5FA641",
        'pink': "#D485B2",
        'blue': "#4277B6",
        'papaya': "#DF8461",
        'violet': "#463397",
        'manilla': "#E1A11A",
        'plum': "#91218C",
        'lemon': "#E8E948",
        'brown': "#7E1510",
        'lime': "#92AE31",
        'dirt': "#6F340D",
        'crimson': "#D32B1E",
        'olive': "#2B3514"
    },
    name="kelly",
    groups={'accent': ['yellow', 'purple', 'orange', 'aqua', 'red', 'buff', 'green', 'pink', 'blue', 'papaya', 'violet', 'manilla', 'plum', 'lemon', 'brown', 'lime', 'dirt', 'crimson', 'olive']}
)

monokai = ColorPalette(
    colors={
        'background': "#282923",
        'comment': "#74705D",
        'foreground': "#F8F8F2",
        'red': "#F92472",
        'orange': "#FD9622",
        'yellow': "#E7DB74",
        'green': "#A6E22B",
        'blue': "#67D8EF",
        'purple': "#AC80FF"
    },
    name="monokai",
    groups={'accent': ['red', 'blue', 'orange', 'yellow', 'green', 'blue', 'purple']}
)

solarized = ColorPalette(
    colors={
        'base03': "#002B36",
        'base02': "#073642",
        'base01': "#586E75",
        'base00': "#657B83",
        'base0': "#839496",
        'base1': "#93A1A1",
        'base2': "#EEE8D5",
        'base3': "#FDF6E3",
        'yellow': "#B58900",
        'orange': "#CB4B16",
        'red': "#DC322F",
        'magenta': "#D33682",
        'violet': "#6C71C4",
        'blue': "#268BD2",
        'cyan': "#2AA198",
        'green': "#859900"
    },
    name="solarized",
    groups={'accent': ['yellow', 'orange', 'red', 'magenta', 'violet', 'blue', 'cyan', 'green']}
)

bonn = ColorPalette(
    colors={
        'blue': "#005AAA",
        'blue_light': "#4B75B9",
        'blue_lighter': "#8197CD",
        'blue_lightest': "#BAC4E4",
        'yellow': "#FDB812",
        'yellow_light': "#FEC85A",
        'yellow_lighter': "#FED88F",
        'yellow_lightest': "#FEE9C3",
        'gray': "#8A8A7C",
        'gray_light': "#A5A599",
        'gray_lighter': "#C2C2B8",
        'gray_lightest': "#DFDFD9"
    },
    name="bonn",
    groups={
        'main': ['blue', 'yellow', 'gray'],
        'light': ['blue_light', 'yellow_light', 'gray_light'],
        'lighter': ['blue_lighter', 'yellow_lighter', 'gray_lighter'],
        'lightest': ['blue_lightest', 'yellow_lightest', 'gray_lightest'],
    }
)

night = ColorPalette(
    colors={
        'blue': "#263C8B",
        'blue_light': "#4E74A6",
        'yellow_light': "#BDBF78",
        'yellow': "#BFA524",
        'black': "#2E231F"
    },
    name="night"
)

wave = ColorPalette(
    colors={
        'indigo': "#011640",
        'blue': "#2D5873",
        'aqua': "#7BA696",
        'gray': "#BFBA9F",
        'brown': "#BF9663"
    },
    name="wave"
)

scream = ColorPalette(
    colors={
        'blue_light': "#4D7186",
        'blue': "#284253",
        'orange': "#E0542E",
        'yellow': "#F4A720",
        'yellow_dark': "#EF8C12"
    },
    name="scream"
)

tangerine = ColorPalette(
    colors={
        'blue': "#003547",
        'green': "#005E54",
        'yellow': "#C2BB00",
        'red': "#E1523D",
        'orange': "#ED8B16"
    },
    name="tangerine"
)

globe = ColorPalette(
    colors={
        'blue': "#042940",
        'teal': "#005C53",
        'green': "#9FC131",
        'lime': "#DBF227",
        'beige': "#D6D58E"
    },
    name="globe"
)

x11 = ColorPalette(
    colors={
        'aliceblue': "#F0F8FF",
        'antiquewhite': "#FAEBD7",
        'aquamarine': "#7FFFD4",
        'azure': "#F0FFFF",
        'beige': "#F5F5DC",
        'bisque': "#FFE4C4",
        'black': "#000000",
        'blanchedalmond': "#FFEBCD",
        'blue': "#0000FF",
        'blueviolet': "#8A2BE2",
        'brown': "#A52A2A",
        'burlywood': "#DEB887",
        'cadetblue': "#5F9EA0",
        'chartreuse': "#7FFF00",
        'chocolate': "#D2691E",
        'coral': "#FF7F50",
        'cornflowerblue': "#6495ED",
        'cornsilk': "#FFF8DC",
        'crimson': "#DC143C",
        'cyan': "#00FFFF",
        'darkblue': "#00008B",
        'darkcyan': "#008B8B",
        'darkgoldenrod': "#B8860B",
        'darkgray': "#A9A9A9",
        'darkgreen': "#006400",
        'darkkhaki': "#BDB76B",
        'darkmagenta': "#8B008B",
        'darkolivegreen': "#556B2F",
        'darkorange': "#FF8C00",
        'darkorchid': "#9932CC",
        'darkred': "#8B0000",
        'darksalmon': "#E9967A",
        'darkseagreen': "#8FBC8F",
        'darkslateblue': "#483D8B",
        'darkslategray': "#2F4F4F",
        'darkturquoise': "#00CED1",
        'darkviolet': "#9400D3",
        'deeppink': "#FF1493",
        'deepskyblue': "#00BFFF",
        'dimgray': "#696969",
        'dodgerblue': "#1E90FF",
        'firebrick': "#B22222",
        'floralwhite': "#FFFAF0",
        'forestgreen': "#228B22",
        'gainsboro': "#DCDCDC",
        'ghostwhite': "#F8F8FF",
        'gold': "#FFD700",
        'goldenrod': "#DAA520",
        'gray': "#808080",
        'green': "#008000",
        'greenyellow': "#ADFF2F",
        'honeydew': "#F0FFF0",
        'hotpink': "#FF69B4",
        'indianred': "#CD5C5C",
        'indigo': "#4B0082",
        'ivory': "#FFFFF0",
        'khaki': "#F0E68C",
        'lavender': "#E6E6FA",
        'lavenderblush': "#FFF0F5",
        'lawngreen': "#7CFC00",
        'lemonchiffon': "#FFFACD",
        'lightblue': "#ADD8E6",
        'lightcoral': "#F08080",
        'lightcyan': "#E0FFFF",
        'lightgoldenrodyellow': "#FAFAD2",
        'lightgray': "#D3D3D3",
        'lightgreen': "#90EE90",
        'lightpink': "#FFB6C1",
        'lightsalmon': "#FFA07A",
        'lightseagreen': "#20B2AA",
        'lightskyblue': "#87CEFA",
        'lightslategray': "#778899",
        'lightsteelblue': "#B0C4DE",
        'lightyellow': "#FFFFE0",
        'lime': "#00FF00",
        'limegreen': "#32CD32",
        'linen': "#FAF0E6",
        'magenta': "#FF00FF",
        'maroon': "#800000",
        'mediumaquamarine': "#66CDAA",
        'mediumblue': "#0000CD",
        'mediumorchid': "#BA55D3",
        'mediumpurple': "#9370DB",
        'mediumseagreen': "#3CB371",
        'mediumslateblue': "#7B68EE",
        'mediumspringgreen': "#00FA9A",
        'mediumturquoise': "#48D1CC",
        'mediumvioletred': "#C71585",
        'midnightblue': "#191970",
        'mintcream': "#F5FFFA",
        'mistyrose': "#FFE4E1",
        'moccasin': "#FFE4B5",
        'navajowhite': "#FFDEAD",
        'navy': "#000080",
        'oldlace': "#FDF5E6",
        'olive': "#808000",
        'olivedrab': "#6B8E23",
        'orange': "#FFA500",
        'orangered': "#FF4500",
        'orchid': "#DA70D6",
        'palegoldenrod': "#EEE8AA",
        'palegreen': "#98FB98",
        'paleturquoise': "#AFEEEE",
        'palevioletred': "#DB7093",
        'papayawhip': "#FFEFD5",
        'peachpuff': "#FFDAB9",
        'peru': "#CD853F",
        'pink': "#FFC0CB",
        'plum': "#DDA0DD",
        'powderblue': "#B0E0E6",
        'purple': "#800080",
        'rebeccapurple': "#663399",
        'red': "#FF0000",
        'rosybrown': "#BC8F8F",
        'royalblue': "#4169E1",
        'saddlebrown': "#8B4513",
        'salmon': "#FA8072",
        'sandybrown': "#F4A460",
        'seagreen': "#2E8B57",
        'seashell': "#FFF5EE",
        'sienna': "#A0522D",
        'silver': "#C0C0C0",
        'skyblue': "#87CEEB",
        'slateblue': "#6A5ACD",
        'slategray': "#708090",
        'snow': "#FFFAFA",
        'springgreen': "#00FF7F",
        'steelblue': "#4682B4",
        'tan': "#D2B48C",
        'teal': "#008080",
        'thistle': "#D8BFD8",
        'tomato': "#FF6347",
        'turquoise': "#40E0D0",
        'violet': "#EE82EE",
        'wheat': "#F5DEB3",
        'white': "#FFFFFF",
        'whitesmoke': "#F5F5F5",
        'yellow': "#FFFF00",
        'yellowgreen': "#9ACD32"
    },
    name="x11"
)
