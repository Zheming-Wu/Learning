# wxauto 在 https://github.com/cluic/wxauto 获取
from wxauto import WeChat
 
# 获取当前微信客户端
wx = WeChat()
 
# 获取会话列表
wx.GetSessionList()

"""
# 输出当前聊天窗口聊天消息
msgs = wx.GetAllMessage
for msg in msgs:
    print('%s : %s'%(msg[0], msg[1]))
## 获取更多聊天记录
wx.LoadMoreMessage()
msgs = wx.GetAllMessage
for msg in msgs:
    print('%s : %s'%(msg[0], msg[1]))
"""
 
# 向某人发送消息（以`张琛萌`为例）
msg1 = '老婆晚安安啦！！！！！！'
msg2 = 'muaaaaaaaaaaaaaaaa~!!!!!!'
msg3 = '爱你！！！！！！！！！'
who = '张琛萌'
wx.ChatWith(who)  # 打开`张琛萌`聊天窗口
wx.SendMsg(msg1)  # 向`张琛萌`发送消息
wx.SendMsg(msg2)  # 向`张琛萌`发送消息
wx.SendMsg(msg3)  # 向`张琛萌`发送消息
 
# 向某人发送文件（以`张琛萌`为例，发送不同类型文件）
file1 = '亲亲.gif'
file2 = '爱你.gif'
file3 = '亲亲.jfif'
file4 = '永远爱你.jfif'
file5 = '抱抱.jfif'
file6 = '贴贴.jfif'
who = '张琛萌'
wx.ChatWith(who)  # 打开`张琛萌`聊天窗口
wx.SendFiles(file1, file2, file3, file4, file5, file6)  # 向`张琛萌`发送上述三个文件

msg4 = '么么哒！'
wx.ChatWith(who)  
wx.SendMsg(msg4)  

file7 = '爱你.gif'
wx.ChatWith(who)  
wx.SendFiles(file7)  

msg5 = 'chu~~~~~~~~~~~'
wx.ChatWith(who)  
wx.SendMsg(msg5)  

file8 = '我爱你.jfif'
wx.ChatWith(who)  
wx.SendFiles(file8)  