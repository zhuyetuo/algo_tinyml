#ifndef __USER_PERIPH_SETUP_H__
#define __USER_PERIPH_SETUP_H__

/* 起 BLE 之前必须调一次：设 BD 地址、起日志、设低功耗模式。 */
void app_periph_init(void);

#endif
