from util import *
import pickle

with open("model.pkl", "rb") as f:
    clf = pickle.load(f)


def classify(text):
    text = transform_text(text)
    cls = clf.predict([text])
    if cls[0] == 1:
        return "Positive"
    else:
        return "Negative"


if __name__ == "__main__":
    text = "Nhân viên thái độ kém, chỗ này không nên đến"
    # text = "Đồ ăn khá ngon, nhân viên thái độ tốt, nếu để đánh giá thì chỗ này được 7 điểm"
    print(classify(text))
