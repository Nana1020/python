import dlib
from scipy.spatial import distance
import cv2
from imutils import face_utils


def eye_aspect_ratio(eye):
    # eye 就是眼部特征点的数组
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('libs/shape_predictor_68_face_landmarks.dat')

# 设置眼睛纵横比的阈值
EAR_THRESH = 0.3
# 我们假定连续三帧以上EAR的值都小于阈值，才确认产生了眨眼操作
EAR_CONSEC_FRAMES = 3
# 人脸特征点中对应眼镜的那几个特征点的序号
RIGHT_EYE_START = 37 - 1  # 数组列表保持一致
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

frame_counter = 0  # 默认是连续帧计数
blink_counter = 0  # 眨眼的计数

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转化为灰度图像
    rects = detector(gray, 1)  # 人脸检测   放大

    if len(rects) > 0:
        shape = predictor(gray, rects[0])  # 检测特征点
        points = face_utils.shape_to_np(shape)
        leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]  # 取出左眼的特征点
        rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]  # 取出右眼的特征点
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # 求EAR平均值
        ear = (leftEAR + rightEAR) / 2.0

        # 实际判断时以下眼轮廓不是必须的
        # 寻找左右眼轮廓
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        # 绘制轮廓
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)  # -1 位置 1 线的粗细
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)  # -1 位置 1 线的粗细

        # 如果EAR小于阈值 开始计算连续帧
        if ear < EAR_THRESH:
            frame_counter += 1
        else:
            if frame_counter >= EAR_CONSEC_FRAMES:
                print('眨眼检测成功，请进入')
                blink_counter += 1
                break
            frame_counter = 0


    cv2.imshow('window', frame)
    if cv2.waitKey(1) &0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
