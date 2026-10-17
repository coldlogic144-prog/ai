// Pre-loaded MNIST sample test digits for quick evaluation
window.SAMPLE_DIGITS = {
  "0": {
    "digit": 0,
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAA7klEQVR4nGNgGGDAPe3vKXlckqq/f//NRuIzIbFFF6ApZkEw8wLMGBgY7JguHsJi6N/fv3///vv79x1jTLlt///+/fv31b2/f/9iyNnfBemc7GvX8Pt3JpqcwvPff3/f6eRiYJB/9vtzISuGL/aIgJm5v//+VkaXPCkHYcqfgEkywT3MZP4IwmJkYmJqRNHZ8/s3jIlp7E2YpKj9899/n8lhl5zw+/fvu7YYwQcG29QZGBiuH2ZAAbf+/vX0fPL3LziYUKUYGArBwQohJqNLyj+DSj7bq8yFLslg1weRRI5sJOCx7vdadw9UT9AfAABd9X/ld0zomAAAAABJRU5ErkJggg==",
    "label": "MNIST Sample 0"
  },
  "1": {
    "digit": 1,
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAiElEQVR4nGNgGARA7V8unM2ELmn47ylune0fccvpfpnGgNNYda6VuHWeus+NU07h3w0GnMbaM7zGLanL0IXTVMu3Zzlw6nQWuvEDp6T+/zU4TZV4cR2Fz4TMSRA7gVtSnuE9TlMZnvxzwqnTVhy3Pobef2eZcenk8mJY8xeXRtZjG7jwmEsnAAAPgx/Aqy44OgAAAABJRU5ErkJggg==",
    "label": "MNIST Sample 1"
  },
  "2": {
    "digit": 2,
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAA30lEQVR4nGNgoAlgRDBLOPVCGKYfX4xN2cq/YHBLDofc1b4Nf/9WYcqZ/Pp7SYGHge3c3x5MSd/flyQZGBiqfvx1xGKsvBCIvPgXqyQYlH7/e4yLATvw+f73uT0SnwlZ0oSNYeVBHBo3fPs7nweHnOSrvy+VccgxHPv7txeXnN+Pv3txGSp8Ao/Gtr9/1+LSyPDj719Q+KEAFiS20G8GBoaPv1n5GQQLGRj+ln9DlrwEIlY/Fw+HcF+0wlPCOn+Eqj//GDadYThyHJFMylgZGLTDGRjmPWBYdx2XywYYAACfcVM+OPLsAgAAAABJRU5ErkJggg==",
    "label": "MNIST Sample 2"
  },
  "3": {
    "digit": 3,
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAABGklEQVR4nGNgoD9ghFA8spkM8y5gV8LT+Pfv31+LhbBKtv4Fg6du2CRT//6ZmJb99u/3Bi5Mya1/lzMwMNi+/vt3MSuG5P+/2iDK6uDfv0tY0CV3/1UA0+Zv/v4NQwgzgcnrDAwMKfvCwxWWMzCoouuM++un8QPi4r/3Q/hRJflieJQ/QCX//v0SieFm74Wb4dIXwe5DAcwCAupqAgITP//9u9MAVUrECsayuvj373YUOd+73/1hbN7bfz94IEtGff/7xxLGMf39dz/cnwwMDMtqGRjhHD1Ghkuo3tnz9+48NRAr//K3v38norqI5+63v79//vz5E+SZE8Lofonf/gTiz8M14iA+NJnAgARv6gGTW2cf/cQIBHoBANhKhxrlGDE5AAAAAElFTkSuQmCC",
    "label": "MNIST Sample 3"
  },
  "4": {
    "digit": 4,
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAA3ElEQVR4nGNgGLzA6AGU4SYLZTAhJN3ZoQy/bgxJFi8Y64wWN7qko+V8KEtIiwvNRt03N3mgzAN/RNEkV3w3hWn8/xdNMuTTZRiz9+9eVlTJlX+yoCyFF7+cUOX4H/6BMdv+wM1gglDs0itgIsoMV9Ccw3nmohCEJfb3bzZMlAVCfb8bvLWPgYFBR1n+P8N/NJ0Mmqu+/vnz58+L57///OGECTLCpQ2VGRgY1jAsjIaZhgXU//mji2YnAjAyMsC9giH5H8k5SPEJARwMP3BayfDiTT5uyc1oAUsDAAD+TjwdVy1eigAAAABJRU5ErkJggg==",
    "label": "MNIST Sample 4"
  },
  "5": {
    "digit": 5,
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAA5ElEQVR4nGNgoD9gxBARlGNgeFh45dZFTMXeM2/+/fv3+re/f9FllPu+/PkLAwwMDCzIkjL5EPrGVQjNApMQKTiy49fHr9y7rpw8//0rqonc5/76MTAoMMgxYbqDbePfFi4cHuJp/fuSH4ccQ8zf+zJYhJnApBXD+Se4NDK8+vut3hBX8P3/x8Dwb8YJuTtXGbSPo5vRDQ+Xv39frECTZDa9de83TPZPDaYFzh4noLLrsTmr4u/PGcZLcEga/f37d8+fv3+nYJPkXA4y9Nd6bqzeFd/y/O/dBqxSIBA7VQynHH0AAAbzdtB6MqF9AAAAAElFTkSuQmCC",
    "label": "MNIST Sample 5"
  },
  "6": {
    "digit": 6,
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAABGUlEQVR4nGNgoDVgP/N3A6oIE0Ku3+D/WVwaS//ssmDApVOCYc8JnJK8v/fgdI7U3yMYYiwwRg1cyFKG4eItVFUP/yaA6emPPv/5864WRY7r6SOQOWZP//5/tf7h/yfyyJJ5f6YxMEg1/PnzuE2GgWHTHwtkOw0ZbjMw1KT/31d4lYEBxEaWlGJgYFALZ5id/wvMPXcOWZKXkZEhV2BZJszPv5AD4f///wyS/yWhxiQ/Q3Ht0T+FDMLPv1cJgzgnvqCEstS9P4UMDJKn/xzkZfA58RPVmww7/27mYmBg8FGWnfX3C5ocg8zFv0eCLC2bj/75ey0QTY6BQfLGnz9//v7582cO2FowYISzBMJVUuf8Z5h7A0Pf4AEAX5tkke39vv0AAAAASUVORK5CYII=",
    "label": "MNIST Sample 6"
  },
  "7": {
    "digit": 7,
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAwElEQVR4nGNgGDaAEUKFpD77sfTFHeyS9xQYGBg+X4UKPuk6A6ZZINxU/Wtahg4Wj2UZGP68lmR4dAZZJxgIGp4xZWD4ceu6UM40Bhwg+O9FIVxyYi//B0OZTBiS2aLvb+LSaP3znx0uOYbWf7tZcclxnv1hhVNj3b9tOOW8f3+wxCUnfPffMlxyzKf/3VbGJan2758vLjn5B/+KkcMaBbT++2eCS872E7okUtja8DDc/YIiCY1sCLjo/A6XsZgAAGkRPJA1cOMmAAAAAElFTkSuQmCC",
    "label": "MNIST Sample 7"
  },
  "8": {
    "digit": 8,
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAABAElEQVR4nGNgGGSAEYWnzcJwEYckZ3IvC8Pl/wwMx9ac+YxmCOeOf3///oXgq5KoOs2nGDOc3MGwV/H7h2B3hcXxKBq7/v49Lgxlz/57GUSxIMt7vYfSYRCaCS5zn4EhFMJSms3DcBbNQVP/PdVgYGBQm/nv378tYAchAeFj/85zKnm8/vv3QTcvAzoQvPz3+od/f181CGJIMTCkPf379+8TX0xdDG6r/oFBOYaMVNOzf3//7ix2uPz3hyia3LIXf/9+X6THyx799e8/C4Q4C4i4Fc7AcHt3oIKuIQPDs+voxrZ9/PcfYuUTJSRhaMBLR0g6MTAw/l875SMWbwwNAABqRmwOTwsFiAAAAABJRU5ErkJggg==",
    "label": "MNIST Sample 8"
  },
  "9": {
    "digit": 9,
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAA00lEQVR4nGNgGKqAb+pBVlxy0Q/+/hXGISfz+t/fv8uEsEtO+Pvv79+/74rZsMjJf/x7Yeffv3+fS2CR9P93kIEj6fa//ycRJjPBGOz/+xl+zLv9//+3X5iSkQzeDAwMJgwMJ75gGhv294JG6LLfb/+90cKUFHoHcu1OlRt/Z2BxkcuH//8mcjC0/buvjE12Xh8PAwPn+r8LGXCCiL+PcAQTAwMD07K/dbi1Gnz9q4ZbtvjfGk6ckqK3/unh1ir3byluSYZdX7AEEwzw3ffDo5UIAAA+/lIBy6j56wAAAABJRU5ErkJggg==",
    "label": "MNIST Sample 9"
  }
};
