"""Display glue: ST7789 LCD + backlight GPIO (Raspberry Pi only).

The app is Pi-only. The OpenCV sim display mode was removed by design, so this
is the only module that touches luma / RPi.GPIO, and luma/RPi.GPIO are
required at import time.
"""
import RPi.GPIO as GPIO
from luma.core.interface.serial import spi
from luma.lcd.device import st7789
from PIL import Image

from config import (
    WIDTH, HEIGHT, BL_PIN,
    SPI_PORT, SPI_DEVICE, SPI_GPIO_DC, SPI_GPIO_RST,
    SPI_BUS_SPEED_HZ, DISPLAY_ROTATE,
)


class Display:
    def __init__(self) -> None:
        GPIO.setmode(GPIO.BCM)
        # initial=GPIO.HIGH so the backlight starts ON (is_on starts True and
        # main.py's startup set_backlight(True) early-returns) — otherwise the
        # LCD would stay dark until the first real state transition.
        GPIO.setup(BL_PIN, GPIO.OUT, initial=GPIO.HIGH)
        self.is_on = True
        serial = spi(
            port=SPI_PORT,
            device=SPI_DEVICE,
            gpio_DC=SPI_GPIO_DC,
            gpio_RST=SPI_GPIO_RST,
            bus_speed_hz=SPI_BUS_SPEED_HZ,
        )
        self._lcd = st7789(serial, width=WIDTH, height=HEIGHT, rotate=DISPLAY_ROTATE)

    def set_backlight(self, on: bool) -> None:
        if on == self.is_on:
            return
        self.is_on = on
        GPIO.output(BL_PIN, GPIO.HIGH if on else GPIO.LOW)

    def show(self, img) -> None:
        """Display a PIL image."""
        self._lcd.display(img)

    def clear(self) -> None:
        self.show(Image.new("RGB", (WIDTH, HEIGHT), "black"))

    def cleanup(self) -> None:
        GPIO.cleanup()