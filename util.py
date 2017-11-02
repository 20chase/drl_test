import time
import itchat

from selenium import webdriver

def capture(url, save_fn="tensorboard.png"):
    browser = webdriver.Firefox() # Get local session of firefox
    browser.set_window_size(1920, 1080)
    browser.get(url) # Load page
    browser.execute_script("""
    (function () {
      var y = 0;
      var step = 100;
      window.scroll(0, 0);
 
      function f() {
        if (y < document.body.scrollHeight) {
          y += step;
          window.scroll(0, y);
          setTimeout(f, 50);
        } else {
          window.scroll(0, 0);
          document.title += "scroll-done";
        }
      }
 
      setTimeout(f, 1000);
     })();
     """)
 
    for i in range(30):
        if "scroll-done" in browser.title:
            break
        time.sleep(1)
 
    browser.save_screenshot(save_fn)
    browser.close()

def wechat_display():
    @itchat.msg_register([itchat.content.TEXT])
    def chat_trigger(msg):
        if msg['Text'] == u'tensorboard':
            capture("http://127.0.0.1:6006/")
            itchat.send_image('tensorboard.png', 'filehelper')

    itchat.auto_login(hotReload=True)
    itchat.run()