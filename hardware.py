"""Display glue: ST7789 LCD + backlight GPIO, or an OpenCV sim window.

This is the ONLY module that touches luma / RPi.GPIO, and only when
DISPLAY_KIND == "st7789". Sim mode lets you run everything on a dev machine.
"""
import cv2
import numpy as np

from config import (
    DISPLAY_KIND, WIDTH, HEIGHT, BL_PIN,
    SPI_PORT, SPI_DEVICE, SPI_GPIO_DC, SPI_GPIO_RST,
    SPI_BUS_SPEED_HZ, DISPLAY_ROTATE,
)


class Display:
    def __init__(self, kind: str = DISPLAY_KIND) -> None:
        self.kind = kind
        self._lcd = None
        self.is_on = True
        if kind == "st7789":
            import RPi.GPIO as GPIO
            from luma.core.interface.serial import spi
            from luma.lcd.device import st7789
            self._GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(BL_PIN, GPIO.OUT)
            serial = spi(
                port=SPI_PORT,
                device=SPI_DEVICE,
                gpio_DC=SPI_GPIO_DC,
                gpio_RST=SPI_GPIO_RST,
                bus_speed_hz=SPI_BUS_SPEED_HZ,
            )
            self._lcd = st7789(serial, width=WIDTH, height=HEIGHT, rotate=DISPLAY_ROTATE)
        else:
            self._window = "looking-eyes (sim)"
            cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)

    def set_backlight(self, on: bool) -> None:
        if on == self.is_on:
            return
        self.is_on = on
        if self.kind == "st7789":
            self._GPIO.output(BL_PIN, self._GPIO.HIGH if on else self._GPIO.LOW)
        else:
            print(f"[sim] backlight {'ON' if on else 'OFF'}")

    def show(self, img) -> None:
        """Display a PIL image."""
        if self.kind == "st7789":
            self._lcd.display(img)
        else:
            cv2.imshow(self._window, np.array(img.convert("RGB")))
            cv2.waitKey(1)

    def clear(self) -> None:
        from PIL import Image
        self.show(Image.new("RGB", (WIDTH, HEIGHT), "black"))

    def cleanup(self) -> None:
        if self.kind == "st7789":
            self._GPIO.cleanup()
        else:
            cv2.destroyAllWindows()