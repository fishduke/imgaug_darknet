https://prisonbreak.tistory.com/6#comment8704379

imgaug 라는 좋은 라이브러리가 있어 이를 활용해 기존 데이터셋의 불균형을 해소하고자 한다

darknet 플랫폼 자체에서 augmentation 기능을 지원해서 특별히 필요 없을 줄 알았는데 하위 버전의 YOLO를 사용해야하니 cfg 설정의 한계를 느껴 데이터 증강 라이브러리를 따로 찾게 되었다.

아래 코드는 폴더 내 이미지를 검색하고, augmentation_count 설정에 따라 반복하여 이미지와 라벨 데이터를 증강시킨다.

imgaug의 Documentation과 https://junyoung-jamong.github.io/ 님의 글을 참고하여 작성하였다.

https://imgur.com/V9Ahat9

https://imgur.com/UpGDVwI
