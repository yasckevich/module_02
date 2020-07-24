#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind

base = pd.read_csv('stud_math.csv')
base.corr()


# In[2]:


# используемые формулы


def vybros(a):
    median = a.median()
    IQR = a.quantile(0.75) - a.quantile(0.25)
    perc25 = a.quantile(0.25)
    perc75 = a.quantile(0.75)
    s = '25-й перцентиль: {},'.format(perc25), '75-й перцентиль: {},'.format(perc75), "IQR: {}, ".format(
        IQR), "Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR)
    return s


def yesno(x):
    x = x.replace('yes', 1)
    x = x.replace('no', 0)
    return x


# In[3]:


# у нас есть дублирующий столбик studytime, granular и studytime
# один из них можно удалять (так как у них полная обратная корреляция)

base.drop(['studytime, granular'], inplace=True, axis=1)


# In[4]:


pd.DataFrame(base).info()
# во всех столбцах более 85% заполнено, нет ни одного, где было бы много пропусков
# предварительно они все нам подходят


# In[5]:


# 1 school — аббревиатура школы, в которой учится ученик

base.school.describe()
base.school = base.school.astype(str).apply(
    lambda x: None if x.strip() == '' else x)

x = pd.DataFrame(base.school.value_counts())
display(x)
print('{} уникальных значений'.format(len(x)))


# In[6]:


# 2 sex — пол ученика ('F' - женский, 'M' - мужской)

base.sex.describe()

x = pd.DataFrame(base.sex.value_counts())
display(x)
print('{} уникальных значений'.format(len(x)))


# In[7]:


# 3 age — возраст ученика (от 15 до 22)

base.age.describe()

print(vybros(base.age))
print()

# есть выброс,поэтому ограничиваем значения
base.age = base.age[base.age.between(13, 21)]


x = pd.DataFrame(base.age.value_counts())
base.age.hist()
print('{} уникальных значений'.format(len(x)))


# In[8]:


# 4 address — тип адреса ученика ('U' - городской, 'R' - за городом)

base.address.describe()

x = pd.DataFrame(base.address.value_counts())
display(x)
print('{} уникальных значений'.format(len(x)))


# In[9]:


# 5 famsize — размер семьи('LE3' <= 3, 'GT3' >3)

base.famsize.describe()

x = pd.DataFrame(base.famsize.value_counts())
display(x)
print('{} уникальных значений'.format(len(x)))


# In[10]:


# 6 Pstatus — статус совместного жилья родителей ('T' - живут вместе 'A' - раздельно)

base.Pstatus.describe()

x = pd.DataFrame(base.Pstatus.value_counts())
display(x)
print('{} уникальных значений'.format(len(x)))


# In[11]:


# 7 Medu — образование матери (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)

base.Medu.describe()

print(vybros(base.Medu))
# выбросов нет

base.Medu.hist()
x = pd.DataFrame(base.Medu.value_counts())
print('{} уникальных значений'.format(len(x)))


# In[12]:


# 8 Fedu — образование отца (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)

base.Fedu.describe()
# ошибка в оценка - максимальное значение должно быть 4.0, а не 40. убираем ошибочное значение

base.Fedu = base.Fedu[base.Fedu.between(0, 5)]

print(vybros(base.Fedu))
print()

# есть выбросы, уберем их
base.Fedu = base.Fedu[base.Fedu.between(0.5, 4.5)]

base.Fedu.hist()
x = pd.DataFrame(base.Fedu.value_counts())
print('{} уникальных значений'.format(len(x)))


# In[13]:


# 9 Mjob — работа матери ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба,
# 'at_home' - не работает, 'other' - другое)

base.Mjob.describe()

x = pd.DataFrame(base.Mjob.value_counts())
display(x)
print('{} уникальных значений'.format(len(x)))
# очень большое кол-во other, почти половина неизвестно. Столбик не имеет права на жизнь
base.drop(['Mjob'], inplace=True, axis=1)


# In[14]:


# 10 Fjob — работа отца ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба,
# 'at_home' - не работает, 'other' - другое)

base.Fjob.describe()

x = pd.DataFrame(base.Fjob.value_counts())
display(x)
print('{} уникальных значений'.format(len(x)))
# очень большое кол-во other, большая часть неизвестна. Столбик не имеет права на жизнь
base.drop(['Fjob'], inplace=True, axis=1)


# In[15]:


# 11 reason — причина выбора школы ('home' - близость к дому, 'reputation' - репутация школы, 'course' -
# образовательная программа, 'other' - другое)

base.reason.describe()

base.reason = base.reason.astype(str).apply(
    lambda x: None if x.strip() == '' else x)
print()

x = pd.DataFrame(base.reason.value_counts())
display(x)
print('{} уникальных значений'.format(len(x)))


# In[16]:


# 12 guardian — опекун ('mother' - мать, 'father' - отец, 'other' - другое)

base.guardian.describe()

x = pd.DataFrame(base.guardian.value_counts())
display(x)
print('{} уникальных значений'.format(len(x)))


# In[17]:


# 13 traveltime — время в пути до школы (1 - <15 мин., 2 - 15-30 мин., 3 - 30-60 мин., 4 - >60 мин.)

base.traveltime.describe()

print(vybros(base.traveltime))
print()

# есть выбросы, уберем их
base.traveltime = base.traveltime[base.traveltime.between(0.0, 3.5)]
base.traveltime.hist()
x = pd.DataFrame(base.traveltime.value_counts())
print('{} уникальных значений'.format(len(x)))


# In[18]:


# 14 studytime — время на учёбу помимо школы в неделю (1 - <2 часов, 2 - 2-5 часов, 3 - 5-10 часов, 4 - >10 часов)

base.studytime.describe()

print(vybros(base.studytime))
print()

# есть выбросы, уберем их
base.studytime = base.studytime[base.studytime.between(0.0, 3.5)]

base.studytime.hist()
x = pd.DataFrame(base.studytime.value_counts())
print('{} уникальных значений'.format(len(x)))


# In[19]:


# 15 failures — количество внеучебных неудач (n, если 1<=n<=3, иначе 0)

base.failures.describe()

print(vybros(base.failures))
print()

# есть выбросы, уберем их
base.failures = base.failures[base.failures.between(0, 0)]

base.failures.hist()
x = pd.DataFrame(base.failures.value_counts())
print('{} уникальных значений'.format(len(x)))

# самые неоднозначный столбик. Показатели выброса не дают возможности для анализа
# а если неудач было больше 3? например, 4? То тогда это также приравнивается к нулю - а, значит, не имеет смысла
base.drop(['failures'], inplace=True, axis=1)


# In[21]:


# 16 schoolsup — дополнительная образовательная поддержка (yes или no)

base.schoolsup.describe()

# переведем yes/no в 1/0, чтобы посчитать корреляцию
base.schoolsup = yesno(base.schoolsup)

print(vybros(base.schoolsup))
print()

# есть выбросы, уберем их
base.schoolsup = base.schoolsup[base.schoolsup.between(0, 0)]

base.schoolsup.hist()
x = pd.DataFrame(base.schoolsup.value_counts())
print('{} уникальных значений'.format(len(x)))

# данный столбик после обработке не носит никакой информативной функции
base.drop(['schoolsup'], inplace=True, axis=1)


# In[22]:


# 17 famsup — семейная образовательная поддержка (yes или no)

base.famsup.describe()

# переведем yes/no в 1/0, чтобы посчитать корреляцию
base.famsup = yesno(base.famsup)

print(vybros(base.famsup))
print()

# выбросов нет

base.famsup.hist()
x = pd.DataFrame(base.famsup.value_counts())
print('{} уникальных значений'.format(len(x)))


# In[23]:


# 18 paid — дополнительные платные занятия по математике (yes или no)

base.paid.describe()

# переведем yes/no в 1/0, чтобы посчитать корреляцию
base.paid = yesno(base.paid)

print(vybros(base.paid))
print()

# выбросов нет

base.paid.hist()
x = pd.DataFrame(base.paid.value_counts())
print('{} уникальных значений'.format(len(x)))


# In[24]:


# 19 activities — дополнительные внеучебные занятия (yes или no)

base.activities.describe()

# переведем yes/no в 1/0, чтобы посчитать корреляцию
base.activities = yesno(base.activities)

print(vybros(base.activities))
print()

# выбросов нет

base.activities.hist()
x = pd.DataFrame(base.activities.value_counts())
print('{} уникальных значений'.format(len(x)))


# In[25]:


# 20 nursery — посещал детский сад (yes или no)

base.nursery.describe()

# переведем yes/no в 1/0, чтобы посчитать корреляцию
base.nursery = yesno(base.nursery)

print(vybros(base.nursery))
print()

# есть выбросы, уберем их
base.nursery = base.nursery[base.nursery.between(1, 1)]

base.nursery.hist()
x = pd.DataFrame(base.nursery.value_counts())
print('{} уникальных значений'.format(len(x)))

# данный столбик после обработке не носит никакой информативной функции
base.drop(['nursery'], inplace=True, axis=1)


# In[26]:


# 21 higher — хочет получить высшее образование (yes или no)

base.higher.describe()

# переведем yes/no в 1/0, чтобы посчитать корреляцию
base.higher = yesno(base.higher)

print(vybros(base.higher))
print()

# есть выбросы, уберем их
base.higher = base.higher[base.higher.between(1, 1)]

base.higher.hist()
x = pd.DataFrame(base.higher.value_counts())
print('{} уникальных значений'.format(len(x)))

# данный столбик после обработке не носит никакой информативной функции
base.drop(['higher'], inplace=True, axis=1)


# In[27]:


# 22 internet — наличие интернета дома (yes или no)

base.internet.describe()

# переведем yes/no в 1/0, чтобы посчитать корреляцию
base.internet = yesno(base.internet)

print(vybros(base.internet))
print()

# есть выбросы, уберем их
base.internet = base.internet[base.internet.between(1, 1)]

base.internet.hist()
x = pd.DataFrame(base.internet.value_counts())
print('{} уникальных значений'.format(len(x)))

# данный столбик после обработке не носит никакой информативной функции
base.drop(['internet'], inplace=True, axis=1)


# In[28]:


# 23 romantic — в романтических отношениях (yes или no)

base.romantic.describe()

# переведем yes/no в 1/0, чтобы посчитать корреляцию
base.romantic = yesno(base.romantic)

print(vybros(base.romantic))
print()

# выбросов нет

base.romantic.hist()
x = pd.DataFrame(base.romantic.value_counts())
print('{} уникальных значений'.format(len(x)))


# In[29]:


# 24 famrel — семейные отношения (от 1 - очень плохо до 5 - очень хорошо)

base.famrel.describe()

# ошибка в оценка - минимальное значение должно быть 0, а не -1. убираем ошибочное значение
base.famrel = base.famrel[base.famrel.between(1, 5)]

print(vybros(base.famrel))
print()

# уберем выбросы
base.famrel = base.famrel[base.famrel.between(2.5, 6.5)]

base.famrel.hist()
x = pd.DataFrame(base.famrel.value_counts())
print('{} уникальных значений'.format(len(x)))


# In[30]:


# 25 freetime — свободное время после школы (от 1 - очень мало до 5 - очень мого)

base.freetime.describe()

print(vybros(base.freetime))
print()

# уберем выбросы
base.freetime = base.freetime[base.freetime.between(1.5, 5.5)]

base.freetime.hist()
x = pd.DataFrame(base.freetime.value_counts())
print('{} уникальных значений'.format(len(x)))


# In[31]:


# 26 goout — проведение времени с друзьями (от 1 - очень мало до 5 - очень много)

base.goout.describe()

print(vybros(base.goout))
print()
# выбросов нет

base.goout.hist()
x = pd.DataFrame(base.goout.value_counts())
print('{} уникальных значений'.format(len(x)))


# In[32]:


# 27 health — текущее состояние здоровья (от 1 - очень плохо до 5 - очень хорошо)

base.goout.describe()

print(vybros(base.health))
print()
# выбросов нет

base.health.hist()
x = pd.DataFrame(base.health.value_counts())
print('{} уникальных значений'.format(len(x)))


# In[33]:


# 28 absences — количество пропущенных занятий

base.absences.describe()

print(vybros(base.absences))
print()

# Есть выбросы, уберем их (отрицательные числа не учитываем)
base.absences = base.absences[base.absences.between(0, 20)]

base.absences.hist()
x = pd.DataFrame(base.absences.value_counts())
print('{} уникальных значений'.format(len(x)))


# In[34]:


# 29 score — баллы по госэкзамену по математике

# ее, мы добрались до последнего столбика!

base.score.describe()

print(vybros(base.score))
print()
# Выбросов нет

base.score.hist()
x = pd.DataFrame(base.score.value_counts())
print('{} уникальных значений'.format(len(x)))
# все ок


# In[35]:


pd.DataFrame(base).info()


# In[36]:


# по столбикам с числовыми данным можно посмотреть взаимосвязь через корреляцию
df = base.corr()
a = df['score']
display(a.sort_values())

# наибольшую связь показывают значения:
# образование родителей оказаывает наибольшее влиние (чем выше степень - тем лучше результат у ребенка)
# возраст (чем старше, тем хуже. Скорее всего, потому что чем старше, тем больше перерыв между пройденным материалом и самим экзаменом в случае повторных экзаменов)
# романтические отношения отказывают не лучшее влияние на учебу
# проведение времени с друзьями и время за учебой(меньше времени остается на учебу и результаты падают)


# In[37]:


# по столбикам со строковыми данным можно посмотреть взаимосвязь через графики


def get_boxplot(column):
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.boxplot(x=column, y='score',
                data=base.loc[base.loc[:, column].isin(
                    base.loc[:, column].value_counts().index[:10])],
                ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Boxplot for ' + column)
    plt.show()


for col in ['school', 'sex', 'address', 'famsize', 'Pstatus', 'reason', 'guardian']:
    get_boxplot(col)

# по строковым данным выводы сдедующие:
# мальчики в большей степени показывают хорошие результаты чем девочки
# ребята, которые живут в городе, имеют больше шансов на хорошую сдачу экзаменов. Возможно, в городе есть доп опции для подготовки

