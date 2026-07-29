from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


BASE = Path(__file__).resolve().parent
OUT = BASE / "output" / "dilation_125.gif"
WIDTH, HEIGHT = 900, 680
GRID_X, GRID_Y, CELL = 54, 142, 30
GRID_N = 17
CENTER = GRID_N // 2
SCAN_CENTERS = [
    (5, 5),
    (5, 8),
    (5, 11),
    (8, 11),
    (8, 8),
    (8, 5),
    (11, 5),
    (11, 8),
    (11, 11),
]

COLORS = {
    1: ((246, 166, 35), (194, 116, 8)),
    2: ((56, 169, 220), (24, 108, 153)),
    5: ((47, 181, 145), (21, 119, 94)),
}


def get_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TITLE = get_font(28)
BODY = get_font(20)
SMALL = get_font(16)
TINY = get_font(12)


def offsets(dilation, center_row, center_col):
    return {
        (center_row + row * dilation, center_col + col * dilation)
        for row in (-1, 0, 1)
        for col in (-1, 0, 1)
    }


def draw_centered(draw, text, y, font, fill):
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((WIDTH - box[2] + box[0]) / 2, y), text, font=font, fill=fill)


def draw_frame(stage, scan_step):
    image = Image.new("RGB", (WIDTH, HEIGHT), (250, 251, 253))
    draw = ImageDraw.Draw(image)
    draw_centered(
        draw, "Stacked dilated convolutions: d=1 -> d=2 -> d=5", 23, TITLE, (30, 42, 54)
    )
    draw_centered(draw, "3x3 kernel, stride=1, scanning on the input grid", 66, BODY, (91, 103, 115))

    active_rates = [1, 2, 5][:stage]
    center_row, center_col = SCAN_CENTERS[scan_step]
    points_by_rate = {
        rate: offsets(rate, center_row, center_col) for rate in active_rates
    }

    draw.text(
        (GRID_X, 108),
        "input grid: highlighted cells are the sampled positions of the current kernel center",
        font=BODY,
        fill=(45, 55, 65),
    )
    for row in range(GRID_N):
        for col in range(GRID_N):
            x0 = GRID_X + col * CELL
            y0 = GRID_Y + row * CELL
            x1, y1 = x0 + CELL - 1, y0 + CELL + CELL - CELL - 1
            point_rates = [
                rate for rate in active_rates if (row, col) in points_by_rate[rate]
            ]
            if len(point_rates) > 1:
                fill = (164, 125, 221)
                outline = (111, 71, 161)
            elif point_rates:
                fill, outline = COLORS[point_rates[0]]
            else:
                fill, outline = (239, 243, 247), (195, 204, 212)
            draw.rectangle(
                (x0, y0, x1, y1),
                fill=fill,
                outline=outline,
                width=2 if point_rates else 1,
            )
            if (row, col) == (center_row, center_col):
                draw.text((x0 + 9, y0 + 7), "O", font=SMALL, fill=(255, 255, 255))

    if active_rates:
        field = 1 + sum(2 * rate for rate in active_rates)
        half = field // 2
        field_row0 = max(0, center_row - half)
        field_col0 = max(0, center_col - half)
        field_row1 = min(GRID_N - 1, center_row + half)
        field_col1 = min(GRID_N - 1, center_col + half)
        x0 = GRID_X + field_col0 * CELL
        y0 = GRID_Y + field_row0 * CELL
        x1 = GRID_X + (field_col1 + 1) * CELL - 1
        y1 = GRID_Y + (field_row1 + 1) * CELL - 1
        draw.rectangle((x0 - 3, y0 - 3, x1, y1), outline=(89, 70, 120), width=3)
    else:
        field = 1

    panel_x = 610
    draw.line((panel_x - 28, 108, panel_x - 28, 600), fill=(210, 216, 222), width=2)
    draw.text((panel_x, 125), "layers", font=BODY, fill=(45, 55, 65))
    y = 174
    for rate in (1, 2, 5):
        fill, outline = COLORS[rate]
        enabled = rate in active_rates
        draw.rectangle(
            (panel_x, y + 3, panel_x + 24, y + 27),
            fill=fill if enabled else (239, 243, 247),
            outline=outline,
        )
        draw.text(
            (panel_x + 38, y),
            f"dilation = {rate}",
            font=BODY,
            fill=(45, 55, 65) if enabled else (160, 168, 176),
        )
        y += 48

    draw.text(
        (panel_x, 335),
        f"receptive field: {field}x{field}",
        font=BODY,
        fill=(89, 70, 120),
    )
    draw.text((panel_x, 377), f"scan step: {scan_step + 1}/{len(SCAN_CENTERS)}", font=SMALL, fill=(91, 103, 115))
    draw.text((panel_x, 402), f"kernel center: ({center_row}, {center_col})", font=SMALL, fill=(91, 103, 115))
    draw.text((panel_x, 427), "colored cells = sampled inputs", font=SMALL, fill=(91, 103, 115))

    map_x, map_y, map_cell = panel_x, 480, 18
    draw.text((panel_x, 452), "output positions visited", font=SMALL, fill=(45, 55, 65))
    for row in range(7):
        for col in range(7):
            x0 = map_x + col * map_cell
            y0 = map_y + row * map_cell
            x1 = x0 + map_cell - 1
            y1 = y0 + map_cell - 1
            source_row = row + 5
            source_col = col + 5
            visited = (source_row, source_col) in SCAN_CENTERS[: scan_step + 1]
            is_current = (source_row, source_col) == (center_row, center_col)
            fill = (232, 238, 244)
            outline = (195, 204, 212)
            if visited:
                fill = (206, 227, 255)
                outline = (86, 140, 206)
            if is_current:
                fill = (255, 214, 153)
                outline = (194, 116, 8)
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=outline)

    note_y = 635
    if stage == 3:
        note = "The kernel center moves, but each layer still uses only 9 sampled points."
        note_fill = (89, 70, 120)
    else:
        note = "Watch how the same 3x3 kernel samples different gaps while sliding on the grid."
        note_fill = (91, 103, 115)
    draw.text((GRID_X, note_y), note, font=SMALL, fill=note_fill)
    return image


def main():
    frames = []
    durations = []
    for stage in (1, 2, 3):
        for scan_step in range(len(SCAN_CENTERS)):
            frames.append(draw_frame(stage, scan_step))
            durations.append(140 if scan_step < len(SCAN_CENTERS) - 1 else 600)
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"saved {OUT} ({OUT.stat().st_size} bytes, {len(frames)} frames)")


if __name__ == "__main__":
    main()
