# EA-VTON - Script thuyết trình tuần 12-19/04/2026

Tài liệu này tách riêng khỏi slide.

Mục đích của file này là:
- giúp người trình bày nói mạch lạc hơn;
- không phải đọc nguyên văn slide;
- giữ lời thoại ngắn, dễ hiểu, dễ nói trước nhóm.

Thời lượng gợi ý:
- Bản ngắn: 5-6 phút.
- Bản đầy đủ: 7-9 phút.

Nguyên tắc khi nói:
- Slide chỉ để người nghe nhìn ý chính.
- Phần script này mới là thứ dẫn câu chuyện.
- Không cần nói hết mọi bullet trên slide.
- Mỗi slide chỉ nên có 1 ý chốt.

## Từ điển thuật ngữ nhanh

Phần này để người trình bày tự nhớ cách giải thích các từ chuyên môn bằng ngôn ngữ đơn giản.

### Reweighting / gán lại trọng số

Không đổi dữ liệu cũ, nhưng cho những mẫu giống người dùng mục tiêu hơn có ảnh hưởng lớn hơn khi huấn luyện.

### Density ratio / tỷ lệ phân bố

Là cách đo một mẫu “giống nhóm mục tiêu đến mức nào” so với dữ liệu nguồn.

Nếu một mẫu phổ biến ở nhóm mục tiêu nhưng hiếm trong dữ liệu nguồn, mẫu đó sẽ được tăng trọng số.

### Copula

Copula là cách giúp mô hình nhìn hai biến đi cùng nhau thay vì nhìn riêng lẻ.

Trong bài toán này, nó giúp nhìn chiều cao và cân nặng như một cặp có liên hệ, thay vì tách rời từng biến.

### PSIS

PSIS là bước làm mềm các trọng số quá lớn.

Hiểu rất đơn giản: nếu vài mẫu bị đẩy trọng số quá cao, mô hình sẽ dễ học lệch. PSIS giúp giảm tình trạng đó để quá trình huấn luyện ổn định hơn.

### ESS

ESS là số mẫu hiệu dụng sau khi đã gán trọng số.

Nếu ESS thấp, nghĩa là chỉ còn một số ít mẫu đang “gánh” phần lớn việc học. Khi đó mô hình dễ thiếu ổn định.

### GBM

GBM là viết tắt của Gradient Boosting Machine.

Đây là một mô hình rất mạnh cho dữ liệu dạng bảng. Nó không học bằng một cây quyết định duy nhất, mà học bằng nhiều cây nhỏ nối tiếp nhau. Mỗi cây mới sẽ cố sửa phần mà các cây trước làm chưa tốt.

Nói theo cách dễ hiểu: GBM giống như sửa bài nhiều vòng. Vòng sau nhìn lỗi của vòng trước rồi vá tiếp. Nhờ vậy, nó thường cho kết quả rất ổn trên dữ liệu có ít đặc trưng nhưng có ý nghĩa rõ ràng như chiều cao, cân nặng, BMI, tuổi.

### MLP

MLP là một mạng neural cơ bản.

Nó linh hoạt hơn GBM, nhưng với dữ liệu bảng nhỏ và vừa thì không phải lúc nào cũng vượt được GBM.

### Within-1

Đây là chỉ số “đúng hoặc chỉ lệch 1 size”.

Ví dụ, nếu nhãn thật là M mà mô hình dự đoán S hoặc L thì vẫn được xem là chấp nhận được.

### Bias

Bias ở đây là xu hướng dự đoán lệch về một phía.

Ví dụ, nếu mô hình hay đoán nhỏ hơn thực tế thì bias đang âm theo hướng under-size.

## Vì sao đi theo hướng này thay vì hướng khác

Phần này có thể dùng khi cần giải thích sâu hơn, hoặc khi bị hỏi trong lúc trình bày.

### 1. Tốt hơn rule-based ở chỗ nào

Rule-based có thể triển khai nhanh, nhưng về lâu dài rất khó tinh chỉnh.

Khi dữ liệu thay đổi hoặc nhóm người dùng thay đổi, rule-based thường phải sửa tay và rất khó đo xem mình vừa sửa tốt hơn thật hay chỉ là hợp với một vài trường hợp.

Trong khi đó, hướng hiện tại học từ dữ liệu, có thể đánh giá bằng số liệu rõ ràng và dễ so sánh giữa các phiên bản.

### 2. Tốt hơn cách gán trọng số độc lập ở chỗ nào

Cách gán trọng số độc lập nhìn chiều cao và cân nặng như hai thứ riêng rẽ.

Nhưng trong thực tế hai biến này đi cùng nhau. Một người cao hơn thường cũng có xu hướng nặng hơn, nên nếu tách rời hoàn toàn thì mô hình dễ đánh giá sai độ “giống” của mẫu.

Copula tốt hơn ở điểm nó nhìn được mối liên hệ này. Kết quả hiện tại cũng cho thấy hướng copula + PSIS có ESS tốt hơn rõ rệt so với cách gán trọng số độc lập.

### 3. Tốt hơn việc dùng trọng số thô không làm mượt ở chỗ nào

Nếu giữ nguyên trọng số thô, một số mẫu hiếm có thể bị đẩy lên quá mạnh, khiến mô hình học lệch.

PSIS giúp làm mềm phần đuôi này, nên dữ liệu huấn luyện ổn định hơn. Trong tài liệu nội bộ của repo, phần ablation cũng dùng đúng câu hỏi này để chứng minh rằng bỏ PSIS ra thì độ ổn định giảm.

### 4. Tốt hơn việc nhảy thẳng sang neural ở chỗ nào

Neural không phải lúc nào cũng tốt hơn mô hình đơn giản, đặc biệt với dữ liệu bảng.

Trong repo hiện tại, GBM được giữ làm baseline mạnh vì:
- đây là loại mô hình thường rất hiệu quả với dữ liệu bảng ít đặc trưng;
- thời gian huấn luyện ngắn;
- dễ kiểm tra và dễ giải thích hơn;
- và quan trọng nhất là kết quả hiện tại ổn định hơn các biến thể neural đã thử.

Nói ngắn gọn, nhóm không loại neural, nhưng chưa có đủ bằng chứng để bỏ GBM và chuyển hẳn sang neural.

### 5. Tốt hơn việc chờ thêm dữ liệu ở chỗ nào

Thu thập thêm dữ liệu là hướng tốt về dài hạn, nhưng không giải quyết được nhu cầu ngắn hạn.

Hướng hiện tại cho phép tận dụng dữ liệu sẵn có ngay bây giờ, đồng thời vẫn để mở cửa cho việc ghép thêm dữ liệu mới về sau.

## Ghi chú tham khảo: Lighter-X

Paper Lighter-X không phải là phương pháp mà dự án này đang triển khai trực tiếp.

Link paper:
- https://www.vldb.org/pvldb/vol18/p3721-zheng.pdf

Tuy nhiên, paper này hữu ích ở tư duy thiết kế hệ thống:
- tác giả nhấn mạnh tính mô-đun và khả năng cắm vào mô hình sẵn có;
- dùng hướng “tách riêng phần lan truyền và phần huấn luyện” để giảm chi phí;
- giảm độ phức tạp tham số từ `O(n × d)` xuống `O(h × d)` với `h << n`;
- trên đồ thị lớn, họ báo cáo kết quả tương đương hoặc tốt hơn với số tham số nhỏ hơn rất nhiều, thậm chí có chỗ chỉ cần khoảng 1% tham số so với LightGCN.

Điểm em rút ra để dùng cho bài toán của mình không phải là “copy Lighter-X”, mà là:
- nếu có thể tách bài toán thành các phần rõ ràng thì sẽ dễ tối ưu hơn;
- không phải lúc nào mô hình lớn hơn, nặng hơn cũng là hướng tốt hơn;
- hướng mô-đun, dễ thay thế và dễ benchmark thường thực tế hơn khi cần triển khai thật.

## Các nghiên cứu gần tinh thần Lighter-X

Phần này dùng để trả lời khi có người hỏi: “Ngoài Lighter-X thì còn paper nào cùng họ tư duy không?”

### 1. LightGCN

Link:
- https://arxiv.org/abs/2002.02126

Ý chính:
- LightGCN là một trong những paper quan trọng nhất của nhánh graph recommendation “nhẹ”.
- Ý tưởng chính là bỏ bớt các phần không cần thiết của GCN truyền thống, chỉ giữ lại phần lan truyền thông tin lân cận.

Liên hệ với bài toán của mình:
- Giống ở tư duy tối giản mô hình để dễ train và dễ dùng hơn.
- Khác ở chỗ LightGCN vẫn là graph recommender, còn bài toán của mình hiện tại là gợi ý size từ dữ liệu cơ thể dạng bảng.

### 2. UltraGCN

Link:
- https://arxiv.org/abs/2110.15114

Ý chính:
- UltraGCN đi xa hơn LightGCN bằng cách bỏ hẳn message passing tường minh và thay bằng ràng buộc xấp xỉ.
- Theo phần giới thiệu của paper, mô hình này hướng tới train nhanh hơn rất nhiều trên bài toán recommendation quy mô lớn.

Liên hệ với bài toán của mình:
- Giống ở mục tiêu làm mô hình nhẹ hơn và dễ deploy hơn.
- Khác ở chỗ UltraGCN giải quyết nút thắt “graph quá nặng”, còn bài toán của mình giải quyết nút thắt “dữ liệu nguồn và dữ liệu mục tiêu khác nhau”.

### 3. JGCF

Link:
- https://arxiv.org/abs/2306.03624

Ý chính:
- JGCF nhìn bài toán graph collaborative filtering theo góc nhìn phổ và dùng đa thức Jacobi để lọc tín hiệu trên graph.
- Paper nhấn mạnh việc xử lý tốt hơn dữ liệu thưa và có tiềm năng tốt hơn cho người dùng cold-start.

Liên hệ với bài toán của mình:
- Giống ở chỗ đều cố gắng khai thác cấu trúc dữ liệu tốt hơn thay vì chỉ tăng độ phức tạp mô hình.
- Khác ở chỗ JGCF là tối ưu tín hiệu trên user-item graph, còn mình đang tối ưu việc thích nghi giữa hai nhóm dân số.

### 4. LightGCL

Link:
- https://arxiv.org/abs/2302.08191

Ý chính:
- LightGCL kết hợp graph recommendation với contrastive learning.
- Điểm quan trọng là paper dùng SVD để tạo augmentation ổn định hơn, nhằm giữ cấu trúc ngữ nghĩa tốt hơn và tăng độ bền trước sparsity và popularity bias.

Liên hệ với bài toán của mình:
- Giống ở tinh thần tăng độ bền và tránh học lệch.
- Khác ở chỗ LightGCL xử lý sparsity/noise trong graph recommendation, còn mình xử lý distribution shift trong size recommendation.

### 5. SVD-GCN

Link:
- https://arxiv.org/abs/2208.12689

Ý chính:
- SVD-GCN giải thích graph convolution theo hướng low-rank và SVD, từ đó đề xuất cách làm đơn giản hơn.
- Paper tập trung vào hai vấn đề gần với Lighter-X là scalability và over-smoothing.

Liên hệ với bài toán của mình:
- Giống ở tư duy: hiểu bản chất rồi đơn giản hóa thay vì tăng thêm độ phức tạp.
- Khác ở chỗ bài toán hiện tại của mình chưa cần giải quyết over-smoothing trên graph.

### 6. Less is More

Link:
- https://arxiv.org/abs/2204.11346

Ý chính:
- Paper này cho rằng không phải toàn bộ tín hiệu trên graph đều hữu ích; nhiều phần thực ra là nhiễu.
- Từ đó, tác giả đề xuất chỉ giữ và reweight những thành phần phổ quan trọng hơn.

Liên hệ với bài toán của mình:
- Đây là paper rất gần về “tinh thần”, vì nó cũng nói rằng không phải cứ nhiều tín hiệu hơn là tốt hơn.
- Trong dự án của mình, tư duy tương tự là: không phải cứ dùng toàn bộ dữ liệu nguồn như nhau là tốt; cần tăng trọng số đúng phần dữ liệu phù hợp hơn với target.

### 7. PPNP / APPNP

Link:
- https://openreview.net/forum?id=H1gL-2A9Ym

Ý chính:
- Dòng paper này nổi bật ở tư duy “predict then propagate”, tức là tách phần học dự đoán và phần lan truyền ra riêng.
- Đây là một trong những tiền đề quan trọng cho các hướng decoupled propagation về sau.

Liên hệ với bài toán của mình:
- Giống ở cách nghĩ: tách bài toán thành các khối rõ ràng để dễ tối ưu và dễ thay thế.
- Đây cũng là lý do tại sao trong hệ thống hiện tại mình tách riêng phần ước lượng số đo, phần gợi ý size và phần thử đồ.

## Chốt lại phần related work

Nếu cần nói ngắn gọn trước nhóm, có thể chốt thế này:

Có khá nhiều nghiên cứu cùng tinh thần với Lighter-X, ví dụ LightGCN, UltraGCN, JGCF, LightGCL, SVD-GCN hay Less is More. Điểm chung của các hướng này là đều cố giảm độ nặng của graph recommender, tách bài toán thành các phần rõ hơn, hoặc chỉ giữ những tín hiệu thực sự quan trọng.

Điều em học từ các paper đó không phải là bê nguyên mô hình về dùng, mà là học tư duy thiết kế: phải biết bài toán khó nhất của mình là gì, rồi chọn cách làm gọn, đo được và gắn vào hệ thống thật. Với EA-VTON hiện tại, bài toán khó nhất là sai lệch phân bố giữa dữ liệu nguồn và nhóm người dùng mục tiêu, nên copula + PSIS + GBM vẫn là hướng sát bài toán hơn.

## Mở đầu

Hôm nay em báo cáo phần tiến triển của EA-VTON trong tuần 12 đến 19 tháng 4.

Trọng tâm của tuần này không phải là thêm thật nhiều thử nghiệm mới, mà là gom những gì đang có thành một quy trình rõ ràng hơn. Tức là từ chỗ chạy nhiều thí nghiệm rời rạc, bây giờ mình muốn có một cách làm có thể giải thích được, đo lại được, và sau đó ghép vào hệ thống thật.

Em sẽ đi theo 3 phần:
- vì sao nhóm chọn hướng này;
- nhóm đang làm theo quy trình nào;
- và hiện tại mình đánh giá kết quả ra sao.

Trước khi đi vào chi tiết, có một điểm rất quan trọng cần nói rõ:

EA-VTON không phải chỉ có một bài toán. Nó có hai nhánh khác nhau.

Nhánh thứ nhất là gợi ý size. Đây là phần nghiên cứu chính, nơi mình xử lý chuyện dữ liệu nguồn và nhóm mục tiêu không giống nhau.

Nhánh thứ hai là virtual try-on. Đây là phần kỹ thuật để tạo ảnh mặc thử. Nó quan trọng về mặt sản phẩm, nhưng không phải là nơi chứa đóng góp nghiên cứu chính của hệ thống.

## Slide 1 - Mở bài

### Ý chính
Tuần này là tuần chuyển từ nghiên cứu rời rạc sang một hệ thống có hai nhánh rõ ràng hơn.

### Script gợi ý
Ở slide đầu tiên, ý em muốn chốt là: mục tiêu của tuần này không phải chỉ để có thêm vài con số mới, mà là để chuẩn hóa cách nhóm làm việc và tách bạch rõ hai phần của hệ thống.

Trước đây mình có nhiều phần làm khá rời nhau: một bên là gợi ý size, một bên là thử đồ ảo, một bên là các script đánh giá. Trong tuần này, nhóm cố gắng gom lại thành một khung chung để nhìn vào là biết phần nào là nghiên cứu, phần nào là kỹ thuật sản phẩm, và mỗi phần phải đo bằng cách nào.

Nói ngắn gọn, tuần này là tuần dọn đường để từ thử nghiệm chuyển sang hệ thống có thể vận hành được, nhưng vận hành theo hai nhánh khác nhau chứ không phải một khối duy nhất.

### Câu chuyển slide
Sau khi nói về mục tiêu chung, em muốn làm rõ ngay kiến trúc tổng thể: hệ thống này gồm hai nhánh nào, và chúng khác nhau ở đâu.

## Slide 2 - Hai nhánh của hệ thống

### Ý chính
Muốn hiểu đúng EA-VTON thì phải tách rõ phần gợi ý size và phần thử đồ ảo.

### Script gợi ý
Ở slide này, em muốn chốt một ý rất quan trọng: size recommendation và VTON có liên quan ở mức sản phẩm, nhưng chúng không phải cùng một bài toán.

Phần size recommendation nhận các thông tin như chiều cao, cân nặng, đặc trưng cơ thể, rồi trả về size, confidence và số đo ước lượng. Đây là nơi nhóm đang làm phần thích nghi dữ liệu nguồn sang nhóm người dùng mục tiêu.

Còn phần VTON nhận ảnh người và ảnh quần áo để dựng ra ảnh mặc thử. Đây là phần thiên về xử lý hình ảnh, tích hợp service và tối ưu trải nghiệm đầu ra.

Nói cách khác, một nhánh trả lời câu hỏi “mặc size nào”, còn nhánh kia trả lời câu hỏi “trông sẽ như thế nào khi mặc”.

Nếu không tách hai phần này ra, mình sẽ rất dễ trộn lẫn cách đánh giá. Ví dụ phần size thì nhìn accuracy, within-1, bias, ESS; còn phần VTON thì nhìn tốc độ, chất lượng ảnh và độ ổn định khuôn mặt.

Ý chốt của slide này là: size recommendation là nhánh nghiên cứu chính; VTON là nhánh kỹ thuật phục vụ sản phẩm.

### Câu chuyển slide
Sau khi tách được hai nhánh, em đi vào nhánh quan trọng nhất trước, tức là phần size recommendation.

## Slide 3 - Size recommendation

### Ý chính
Size recommendation là nơi chứa phần đóng góp phương pháp chính của hệ thống.

### Script gợi ý
Vấn đề cốt lõi ở nhánh này là dữ liệu mình có và nhóm người dùng mục tiêu không giống nhau.

Nếu lấy nguyên mô hình học từ dữ liệu hiện tại rồi áp thẳng cho nhóm mục tiêu, mô hình sẽ dễ lệch size.

Vì vậy, nhóm chọn cách vẫn dùng dữ liệu sẵn có, nhưng gán trọng số cao hơn cho những mẫu giống người dùng mục tiêu hơn.

Sau đó, nhóm huấn luyện và so sánh các mô hình như GBM và MLP để xem hướng nào vừa ổn định, vừa hợp với dữ liệu bảng.

Điểm quan trọng ở đây là nhánh size không chỉ dừng ở việc dự đoán một nhãn M hay L, mà còn phải trả về confidence và số đo ước lượng để đi tiếp vào phần sản phẩm.

Đây là nơi có phần thích nghi miền, phần so sánh mô hình và phần nghiên cứu chính của toàn bộ hệ thống.

### Câu chuyển slide
Sau phần nghiên cứu chính là nhánh còn lại, tức phần VTON và cách hai nhánh gặp nhau trong hệ thống.

## Slide 4 - VTON và serving

### Ý chính
VTON là phần kỹ thuật phục vụ sản phẩm, còn API là nơi nối hai nhánh lại với nhau.

### Script gợi ý
Ở slide này, em muốn nhấn mạnh rằng phần VTON khác phần size recommendation ở bản chất công việc.

Nhánh size chủ yếu xoay quanh dữ liệu, trọng số, mô hình và đánh giá.

Nhánh VTON chủ yếu xoay quanh ảnh đầu vào, service xử lý, dựng ảnh đầu ra và thời gian chạy.

Hai nhánh này gặp nhau ở API cuối. Tức là người dùng gửi một request, hệ thống sẽ gọi phần ước lượng số đo, gọi phần gợi ý size, rồi gọi phần thử đồ.

Nhưng dù chúng gặp nhau ở cùng một endpoint, mình vẫn phải nhớ là logic cải tiến của chúng khác nhau. Phần size có thể cải tiến bằng thay đổi trọng số hay đổi model. Phần VTON lại cải tiến bằng cách thay backend, tối ưu xử lý ảnh hoặc đổi cấu hình chạy.

Ý chốt của slide này là: ở mức hệ thống, hai nhánh hợp lại thành một sản phẩm; ở mức kỹ thuật, chúng vẫn là hai bài toán riêng.

### Câu chuyển slide
Khi đã có quy trình làm rõ ràng, phần tiếp theo là xác định xem mình đánh giá tốt hay chưa bằng cách nào.

## Slide 5 - Khung đánh giá

### Ý chính
Nhóm không nhìn một con số duy nhất, mà đánh giá theo nhiều mặt để tránh chọn nhầm mô hình.

### Script gợi ý
Ở phần gợi ý size, nếu chỉ nhìn đúng hay sai tuyệt đối thì chưa đủ. Vì trong thực tế, lệch một size đôi khi vẫn chấp nhận được, còn lệch nhiều mới là vấn đề lớn.

Nên nhóm dùng thêm chỉ số kiểu “đúng hoặc chỉ lệch 1 size”. Song song với đó, nhóm cũng theo dõi xem mô hình có đang nghiêng hẳn theo một hướng nào không, ví dụ hay dự đoán nhỏ hơn thực tế hoặc lớn hơn thực tế.

Ngoài ra còn có một chỉ số quan trọng là số mẫu hiệu dụng. Có thể hiểu đơn giản là sau khi gán trọng số, dữ liệu còn “đủ khỏe” để học hay không. Nếu chỉ vài mẫu chi phối toàn bộ quá trình học thì dù kết quả có đẹp cũng không đáng tin.

Với phần thử đồ ảo thì logic cũng tương tự. Không thể chỉ nhìn ảnh đẹp hay xấu, mà phải nhìn cùng lúc thời gian chạy, độ ổn định của khuôn mặt, độ ổn định của da, và khả năng dùng làm mốc so sánh cho các vòng sau.

Ý chính của slide này là: nhóm đang cố chọn mô hình theo cách cân bằng, chứ không chạy theo một con số đẹp.

### Câu chuyển slide
Sau khi có khung đánh giá, phần quan trọng nhất là xem kết quả hiện tại đang nói gì.

## Slide 6 - Kết quả hiện tại

### Ý chính
Hướng copula + PSIS đang là lựa chọn tốt nhất ở thời điểm này, nhưng chưa phải kết luận cuối cùng.

### Script gợi ý
Nhìn vào kết quả hiện tại trên nhóm VN, có thể thấy cách làm copula cộng với PSIS cho độ chính xác tốt nhất trong ba cấu hình so sánh.

Nó không phải vượt quá xa, nhưng có hai tín hiệu đáng chú ý.

Tín hiệu thứ nhất là độ chính xác tăng lên.

Tín hiệu thứ hai, và theo em là quan trọng hơn ở giai đoạn này, là số mẫu hiệu dụng tăng mạnh. Điều này cho thấy cách gán trọng số mới ổn hơn và ít bị phụ thuộc vào một vài mẫu hiếm.

Ngoài ra, nếu so giữa các dòng mô hình, thì hiện tại GBM vẫn đều hơn các biến thể neural mà nhóm đã thử.

Tuy vậy, mình chưa nên kết luận quá sớm. Chỉ số đúng hoặc lệch một size chưa tăng đủ mạnh để nói rằng bài toán đã được giải quyết.

Vì thế, cách hợp lý nhất lúc này là lấy `gbm_copula` làm mốc so sánh chính, rồi tiếp tục cải thiện từ đó.

### Câu chuyển slide
Phần size recommendation là một nửa câu chuyện. Nửa còn lại là thử đồ ảo, nơi mình cần một mốc đo thực tế hơn về chất lượng và thời gian.

## Slide 7 - Benchmark VTON

### Ý chính
Với VTON, 20 bước đang là điểm cân bằng hợp lý nhất giữa chất lượng và thời gian.

### Script gợi ý
Ở phần thử đồ ảo, nhóm so sánh ba mức chạy: 10 bước, 20 bước và 30 bước.

Kết quả cho thấy 10 bước thì chạy nhanh hơn, nhưng chất lượng ảnh vẫn còn thiếu chi tiết. 30 bước thì ảnh có tốt hơn, nhưng phần tăng thêm không tương xứng với thời gian phải bỏ ra.

Nên ở thời điểm hiện tại, 20 bước là mốc hợp lý nhất. Nó chưa chắc là mốc tốt nhất mãi mãi, nhưng là mốc đủ cân bằng để nhóm dùng làm chuẩn so sánh cho các vòng sau.

Điều quan trọng ở slide này không phải là 20 bước là “đúng tuyệt đối”, mà là từ bây giờ nhóm đã có một mốc cố định để so sánh công bằng giữa các lần cải tiến.

### Câu chuyển slide
Sau khi đi qua lý do chọn hướng làm, cách triển khai và kết quả hiện tại, em chốt lại bằng 4 ý chính.

## Slide 8 - Kết luận

### Ý chính
Nhóm đã có một hướng chính rõ ràng cho phần size recommendation và một cách đo ổn định hơn cho phần VTON.

### Script gợi ý
Em chốt lại bằng bốn ý ngắn.

Thứ nhất, với phần gợi ý size, ở thời điểm hiện tại hướng copula cộng PSIS cộng GBM là hướng phù hợp nhất để đi tiếp.

Thứ hai, nhóm đã có một quy trình tương đối rõ: mô tả nhóm mục tiêu, gán trọng số, huấn luyện, chọn mô hình, rồi ghép vào hệ thống.

Thứ ba, việc đánh giá bây giờ cũng rõ ràng hơn trước, vì mình không chỉ nhìn độ chính xác mà còn nhìn độ ổn định và khả năng dùng thật.

Thứ tư, việc tách hệ thống thành các service giúp nhóm dễ cải tiến từng phần mà không phải phá vỡ toàn bộ luồng xử lý.

Nếu nói ngắn gọn trong một câu, thì kết quả lớn nhất của tuần này là: nhóm đã bắt đầu chuyển từ “thử xem có chạy được không” sang “xây một cách làm có thể đo, so sánh và triển khai tiếp”.

## Câu chốt cuối

Phần em muốn mọi người nhớ sau buổi này là:

EA-VTON tuần này chưa phải là đã giải xong bài toán, nhưng đã có một hướng đi đủ rõ để tiếp tục làm một cách có hệ thống. Tức là mình đã biết nên đo cái gì, nên giữ cái gì làm mốc, và nên cải thiện từ đâu ở vòng tiếp theo.

## Hỏi đáp gợi ý

### Nếu bị hỏi: “Tại sao không đi thu thập thêm dữ liệu luôn?”

Trả lời gợi ý:

Thu thập thêm dữ liệu chắc chắn là hướng tốt về dài hạn. Nhưng ở giai đoạn hiện tại, nhóm cần một cách làm có thể dùng ngay với dữ liệu sẵn có. Reweighting là cách hợp lý để đi bước đầu, rồi sau đó nếu có thêm dữ liệu thì vẫn ghép được vào quy trình này.

### Nếu bị hỏi: “Tại sao chưa chọn neural là hướng chính?”

Trả lời gợi ý:

Vì ở kết quả hiện tại, GBM đang ổn định hơn. Neural vẫn còn tiềm năng, nhưng chưa cho lợi thế đủ rõ để thay GBM làm mốc chính ngay bây giờ.

### Nếu bị hỏi: “Tại sao không đi theo hướng graph recommender như Lighter-X?”

Trả lời gợi ý:

Lighter-X rất đáng tham khảo về mặt tư duy hệ thống, nhất là ở tính mô-đun, nhẹ và dễ cắm vào pipeline sẵn có.

Nhưng bài toán chính của mình hiện tại không phải là tối ưu một graph recommender cực lớn, mà là xử lý sai lệch phân bố giữa dữ liệu nguồn và nhóm người dùng mục tiêu trong bài toán gợi ý size.

Nói cách khác, vấn đề khó nhất của mình lúc này là “dữ liệu khác nhau”, chứ chưa phải “mô hình graph quá nặng”.

Vì vậy, hướng copula + PSIS + GBM phù hợp hơn cho giai đoạn này: đơn giản hơn, đúng trọng tâm hơn, dễ kiểm chứng hơn, và dễ đưa vào hệ thống thật hơn.

### Nếu bị hỏi: “Tại sao chọn 20 bước cho VTON?”

Trả lời gợi ý:

Vì 10 bước thì nhanh nhưng chất lượng còn yếu, 30 bước thì tốn thời gian nhiều mà lợi ích tăng không tương xứng. 20 bước là mốc cân bằng nhất để dùng làm chuẩn so sánh ở thời điểm hiện tại.
