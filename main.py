import cv2 as cv
import numpy as np
import math


def get_rotate_angle(src_pts, dst_pts):
    # Получаем матрицу афинного преобразования, размерность 2*3
    # Элемент a21 = -scale * cos(alpha)
    # Элемент a21 = scale * sin(alpha)
    # Чтобы найти alpha нужно взять arctan(-a21 / a22)
    M = cv.getAffineTransform(src_pts, dst_pts)

    a21 = M[1][0]
    a22 = M[1][1]

    if a22 == 0:
        alpha = 90,
    else:
        # Возвращает значение в радианах
        alpha = math.atan(-a21 / a22)
        alpha = math.degrees(alpha)

    return alpha


def get_points(f, pts, p):
    center_x = 0
    center_y = 0
    for i in range(p):
        d = f.readline().split(" ")
        x = int(d[0])
        y = int(d[1])
        if i < p - 1:
            pts[0][i][0] = x
            pts[0][i][1] = y
        center_y += y
        center_x += x

    return pts, (center_x / p, center_y / p)


def read_data(name_file):
    f = open(name_file, "r")
    p = int(f.readline())
    src = np.empty((1, 3, 2), dtype='float32')
    dst = np.empty((1, 3, 2), dtype='float32')

    src, center_src = get_points(f, src, p)
    f.readline()
    dst, center = get_points(f, dst, p)

    return src, center_src, dst, center


def generate_test(src, center_src):
    test_alpha = 30

    # Получаем матрицу поворота на заданный угол
    rot_mat = cv.getRotationMatrix2D(center_src, test_alpha, 1)
    # Добавляем к матрице поворота вектор (0,0,1) для того чтобы применить перспективное преобразование
    rot_mat = np.vstack((rot_mat, [0, 0, 1]))
    # Получаем точки после поворота
    dst = cv.perspectiveTransform(src, rot_mat)

    # Делаем растяжение по осям и сдвиг
    for i in range(3):
        dst[0][i][0] *= 10 - 100
        dst[0][i][1] *= 20 + 6

    alpha = get_rotate_angle(src, dst)
    print("test angle = " + str(alpha))
    assert abs(test_alpha - alpha) < 0.1


if __name__ == '__main__':
    # Чтение данных
    src, center_src, dst, center = read_data("test.txt")
    # Генерируем тест, к исходным точкам применяем поворот+растяжение по осям и сдвиг
    # После этого сравниваем ответы
    generate_test(src, center_src)
    angle = get_rotate_angle(src, dst)
    print("rotate angle = " + str(angle))
